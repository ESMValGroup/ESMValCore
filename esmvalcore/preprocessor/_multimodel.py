"""Functions to compute multi-cube statistics."""

import logging
import re
from datetime import datetime
from functools import reduce

import cf_units
import dask
import dask.array as da
import iris
import numpy as np
from iris.util import equalise_attributes

logger = logging.getLogger(__name__)

STATISTIC_MAPPING = {
    'gmean': iris.analysis.GMEAN,  # not lazy in iris
    'hmean': iris.analysis.HMEAN,  # not lazy in iris
    'max': iris.analysis.MAX,
    'median': iris.analysis.MEDIAN,  # not lazy in iris
    'min': iris.analysis.MIN,
    'rms': iris.analysis.RMS,
    'sum': iris.analysis.SUM,
    'mean': iris.analysis.MEAN,
    'std_dev': iris.analysis.STD_DEV,
    'variance': iris.analysis.VARIANCE,
    # The following require extra kwargs,
    # atm this is only supported for percentiles via e.g. `pXX`
    'count': iris.analysis.COUNT,
    'peak': iris.analysis.PEAK,
    'percentile': iris.analysis.PERCENTILE,  # not lazy in iris
    'proportion': iris.analysis.PROPORTION,  # not lazy in iris
    'wpercentile': iris.analysis.WPERCENTILE,  # not lazy in iris
}


def _get_consistent_time_unit(cubes):
    """Return cubes' time unit if consistent, standard calendar otherwise."""
    t_units = [cube.coord('time').units for cube in cubes]
    if len(set(t_units)) == 1:
        return t_units[0]
    return cf_units.Unit("days since 1850-01-01", calendar="standard")


def _unify_time_coordinates(cubes):
    """Make sure all cubes' share the same time coordinate.

    This function extracts the date information from the cube and
    reconstructs the time coordinate, resetting the actual dates to the
    15th of the month or 1st of july for yearly data (consistent with
    `regrid_time`), so that there are no mismatches in the time arrays.

    If cubes have different time units, it will use reset the calendar to
    a default gregorian calendar with unit "days since 1850-01-01".

    Might not work for (sub)daily data, because different calendars may have
    different number of days in the year.
    """
    t_unit = _get_consistent_time_unit(cubes)

    for cube in cubes:
        # Extract date info from cube
        coord = cube.coord('time')
        years = [p.year for p in coord.units.num2date(coord.points)]
        months = [p.month for p in coord.units.num2date(coord.points)]
        days = [p.day for p in coord.units.num2date(coord.points)]

        # Reconstruct default calendar
        if 0 not in np.diff(years):
            # yearly data
            dates = [datetime(year, 7, 1, 0, 0, 0) for year in years]
        elif 0 not in np.diff(months):
            # monthly data
            dates = [
                datetime(year, month, 15, 0, 0, 0)
                for year, month in zip(years, months)
            ]
        elif 0 not in np.diff(days):
            # daily data
            dates = [
                datetime(year, month, day, 0, 0, 0)
                for year, month, day in zip(years, months, days)
            ]
            if coord.units != t_unit:
                logger.warning(
                    "Multimodel encountered (sub)daily data and inconsistent "
                    "time units or calendars. Attempting to continue, but "
                    "might produce unexpected results.")
        else:
            raise ValueError(
                "Multimodel statistics preprocessor currently does not "
                "support sub-daily data.")

        # Update the cubes' time coordinate (both point values and the units!)
        cube.coord('time').points = t_unit.date2num(dates)
        cube.coord('time').units = t_unit
        cube.coord('time').bounds = None
        cube.coord('time').guess_bounds()


def _time_coords_are_aligned(cubes):
    """Return `True` if time coords are aligned."""
    first_time_array = cubes[0].coord('time').points

    for cube in cubes[1:]:
        other_time_array = cube.coord('time').points
        if not np.array_equal(first_time_array, other_time_array):
            return False

    return True


def _subset(cube, time_points):
    """Subset cube to given time range."""
    begin = cube.coord('time').units.num2date(time_points[0])
    end = cube.coord('time').units.num2date(time_points[-1])
    constraint = iris.Constraint(time=lambda cell: begin <= cell.point <= end)
    return cube.extract(constraint)


def _extend(cube, time_points):
    """Extend cube to a specified time range.

    If time points are missing before the start/after the end of the
    time range, cubes for each missing time pointwith masked data will
    be added to pad the time range to match `time_points`. This method
    supports lazy operation.
    """
    cube.coord('time').bounds = None
    cube_points = cube.coord('time').points

    begin = cube_points[0]
    end = cube_points[-1]

    pad_begin = time_points[time_points < begin]
    pad_end = time_points[time_points > end]

    if (len(pad_begin)) == 0 and (len(pad_end) == 0):
        return cube

    template_cube = cube[:1].copy()
    template_cube.data = np.ma.array(da.zeros_like(template_cube.data),
                                     mask=True,
                                     dtype=template_cube.data.dtype)

    cube_list = []

    for time_point in pad_begin:
        new_slice = template_cube.copy()
        new_slice.coord('time').points = float(time_point)
        cube_list.append(new_slice)

    cube_list.append(cube)

    for time_point in pad_end:
        new_slice = template_cube.copy()
        new_slice.coord('time').points = float(time_point)
        cube_list.append(new_slice)

    cube_list = iris.cube.CubeList(cube_list)

    new_cube = cube_list.concatenate_cube()

    return new_cube


def _align(cubes, span):
    """Expand or subset cubes so they share a common time span."""
    _unify_time_coordinates(cubes)

    if _time_coords_are_aligned(cubes):
        return cubes

    all_time_arrays = [cube.coord('time').points for cube in cubes]

    if span == 'overlap':
        common_time_points = reduce(np.intersect1d, all_time_arrays)
        new_cubes = [_subset(cube, common_time_points) for cube in cubes]
    elif span == 'full':
        all_time_points = reduce(np.union1d, all_time_arrays)
        new_cubes = [_extend(cube, all_time_points) for cube in cubes]
    else:
        raise ValueError(f"Invalid argument for span: {span!r}"
                         "Must be one of 'overlap', 'full'.")

    for cube in new_cubes:
        # Make sure bounds exist and are consistent
        cube.coord('time').bounds = None
        cube.coord('time').guess_bounds()

    return new_cubes


def _combine(cubes, dim='new_dim'):
    """Merge iris cubes into a single big cube with new dimension.

    This assumes that all input cubes have the same shape.
    """
    equalise_attributes(cubes)  # in-place

    for i, cube in enumerate(cubes):
        concat_dim = iris.coords.AuxCoord(i, var_name=dim)
        cube.add_aux_coord(concat_dim)

        # Clear some metadata that can cause merge to fail
        # https://scitools-iris.readthedocs.io/en/stable/userguide/
        #    merge_and_concat.html#common-issues-with-merge-and-concatenate

        cube.cell_methods = None

        for coord in cube.coords():
            coord.long_name = None
            coord.attributes = None

    cubes = iris.cube.CubeList(cubes)

    merged_cube = cubes.merge_cube()

    return merged_cube


def rechunk(cube, blocksize='auto'):
    """Rechunk the cube to speed up out-of-memory computation."""

    if blocksize != 'auto':  # auto block size in dask is "128MiB"
        dask.config.set({"array.chunk-size": blocksize})

    new_chunks = {0: -1}  # don't chunk along the multimodel dimension
    if cube.ndim > 1:
        new_chunks[1] = 'auto'  # do chunk along the first subsequent dimension

    cube.data = cube.lazy_data().rechunk(new_chunks)

    logger.debug("Total data size: %s MB", cube.lazy_data().nbytes * 1e-6)
    logger.debug("New chunk block size: %s MB",
                 cube.lazy_data().nbytes / cube.lazy_data().npartitions * 1e-6)
    logger.debug("New chunk configuration: %s", cube.lazy_data())


def _compute(cube, statistic: str, dim: str = 'new_dim'):
    """Compute statistic.

    Parameters
    ----------
    cube : :obj:`iris.cube.Cube`
        Input cube.
    statistic : str
        Name of the statistic to calculate. Must be available via
        :mod:`iris.analysis`.
    dim : str
        Collapse cube along this coordinate.

    Returns
    -------
    :obj:`iris.cube.Cube`
        Collapsed cube.
    """
    statistic = statistic.lower()
    kwargs = {}

    rechunk(cube, blocksize="auto")

    # special cases
    if statistic == 'std':
        logger.warning(
            "Multicube statistics is aligning its behaviour with iris.analysis"
            ". Please consider replacing 'std' with 'std_dev' in your code.")
        statistic = 'std_dev'

    elif re.match(r"^(p\d{1,2})(\.\d*)?$", statistic):
        # percentiles between p0 and p99.99999...
        percentile = float(statistic[1:])
        kwargs['percent'] = percentile
        statistic = 'percentile'

    try:
        operator = STATISTIC_MAPPING[statistic]
    except KeyError as err:
        raise ValueError(
            f'Statistic `{statistic}` not supported by multicube statistics. '
            f'Must be one of {tuple(STATISTIC_MAPPING.keys())}.') from err

    logger.debug('Multicube statistics: computing: %s', statistic)

    # This will always return a masked array
    return cube.collapsed(dim, operator, **kwargs)


def _multicube_statistics(cubes, statistics, span):
    """Compute statistics over multiple cubes.

    Can be used e.g. for ensemble or multi-model statistics.

    Cubes are merged and subsequently collapsed along a new auxiliary
    coordinate. Inconsistent attributes will be removed.
    """
    realize = False
    for cube in cubes:
        # make input cubes lazy for efficient operation on real data
        if not cube.has_lazy_data():
            cube.data = cube.lazy_data()
            realize = True

    # work with copy of cubes to avoid modifying input cubes
    copied_cubes = [cube.copy() for cube in cubes]

    aligned_cubes = _align(copied_cubes, span=span)
    big_cube = _combine(aligned_cubes)
    statistics_cubes = {}
    for statistic in statistics:
        result_cube = _compute(big_cube, statistic)

        # realize data if input cubes are not lazy
        if realize:
            result_cube.data

        statistics_cubes[statistic] = result_cube

    return statistics_cubes


def _multiproduct_statistics(products,
                             statistics,
                             output_products,
                             span=None,
                             keep_input_datasets=None):
    """Compute multi-cube statistics on ESMValCore products.

    Extract cubes from products, calculate multicube statistics and
    assign the resulting output cubes to the output_products.
    """
    cubes = [cube for product in products for cube in product.cubes]
    statistics_cubes = _multicube_statistics(cubes=cubes,
                                             statistics=statistics,
                                             span=span)
    statistics_products = set()
    for statistic, cube in statistics_cubes.items():
        statistics_product = output_products[statistic]
        statistics_product.cubes = [cube]

        for product in products:
            statistics_product.wasderivedfrom(product)

        logger.info("Generated %s", statistics_product)
        statistics_products.add(statistics_product)

    if not keep_input_datasets:
        return statistics_products

    return products | statistics_products


def multi_model_statistics(products,
                           span,
                           statistics,
                           output_products=None,
                           keep_input_datasets=True):
    """Compute multi-model statistics.

    This function computes multi-model statistics on a list of ``products``,
    which can be instances of :py:class:`~iris.cube.Cube` or
    :py:class:`~esmvalcore.preprocessor.PreprocessorFile`.
    The latter is used internally by ESMValCore to store
    workflow and provenance information, and this option should typically be
    ignored.

    Apart from the time coordinate, cubes must have consistent shapes. There
    are two options to combine time coordinates of different lengths, see the
    ``span`` argument.

    Uses the statistical operators in :py:mod:`iris.analysis`, including
    ``mean``, ``median``, ``min``, ``max``, and ``std``. Percentiles are also
    supported and can be specified like ``pXX.YY`` (for percentile ``XX.YY``;
    decimal part optional).

    Notes
    -----
    Some of the operators in :py:mod:`iris.analysis` require additional
    arguments. Except for percentiles, these operators are currently not
    supported.

    Lazy operation is supported for all statistics, except
    ``median``, ``percentile``, ``gmean`` and ``hmean``.

    Parameters
    ----------
    products: list
        Cubes (or products) over which the statistics will be computed.
    statistics: list
        Statistical metrics to be computed, e.g. [``mean``, ``max``]. Choose
        from the operators listed in the iris.analysis package. Percentiles can
        be specified like ``pXX.YY``.
    span: str
        Overlap or full; if overlap, statitstics are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data.
    output_products: dict
        For internal use only. A dict with statistics names as keys and
        preprocessorfiles as values. If products are passed as input, the
        statistics cubes will be assigned to these output products.
    keep_input_datasets: bool
        If True, the output will include the input datasets.
        If False, only the computed statistics will be returned.

    Returns
    -------
    dict
        A dictionary of statistics cubes with statistics' names as keys. (If
        input type is products, then it will return a set of output_products.)

    Raises
    ------
    ValueError
        If span is neither overlap nor full, or if input type is neither cubes
        nor products.
    """
    if all(isinstance(p, iris.cube.Cube) for p in products):
        return _multicube_statistics(
            cubes=products,
            statistics=statistics,
            span=span,
        )
    if all(type(p).__name__ == 'PreprocessorFile' for p in products):
        # Avoid circular input: https://stackoverflow.com/q/16964467
        return _multiproduct_statistics(
            products=products,
            statistics=statistics,
            output_products=output_products,
            span=span,
            keep_input_datasets=keep_input_datasets,
        )
    raise ValueError(
        "Input type for multi_model_statistics not understood. Expected "
        "iris.cube.Cube or esmvalcore.preprocessor.PreprocessorFile, "
        "got {}".format(products))
