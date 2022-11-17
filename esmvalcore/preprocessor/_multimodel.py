"""Statistics across cubes.

This module contains functions to compute statistics across multiple cubes or
products.

Wrapper functions separate esmvalcore internals, operating on products, from
generalized functions that operate on iris cubes. These wrappers support
grouped execution by passing a groupby keyword.
"""
import logging
import re
import warnings
from datetime import datetime
from functools import reduce

import cf_units
import iris
import iris.coord_categorisation
import numpy as np
from iris.util import equalise_attributes

from esmvalcore.iris_helpers import date2num
from esmvalcore.preprocessor import remove_fx_variables

from ._other import _group_products

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

CONCAT_DIM = 'multi-model'


def _resolve_operator(statistic: str):
    """Find the operator corresponding to the statistic."""
    statistic = statistic.lower()
    kwargs = {}

    # special cases
    if statistic == 'std':
        logger.warning(
            "Changing statistics from specified `std` to `std_dev`, "
            "since multimodel statistics is now using the iris.analysis module"
            ", which also uses `std_dev`. Please consider replacing 'std' "
            " with 'std_dev' in your recipe or code.")
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

    return operator, kwargs


def _get_consistent_time_unit(cubes):
    """Return cubes' time unit if consistent, standard calendar otherwise."""
    t_units = [cube.coord('time').units for cube in cubes]
    if len(set(t_units)) == 1:
        return t_units[0]
    return cf_units.Unit("days since 1850-01-01", calendar="standard")


def _unify_time_coordinates(cubes):
    """Make sure all cubes' share the same time coordinate.

    This function extracts the date information from the cube and reconstructs
    the time coordinate, resetting the actual dates to the 15th of the month or
    1st of July for yearly data (consistent with `regrid_time`), so that there
    are no mismatches in the time arrays.

    If cubes have different time units, it will reset the calendar to a
    the "standard" calendar with unit "days since 1850-01-01".

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
        cube.coord('time').points = date2num(dates, t_unit, coord.dtype)
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


def _map_to_new_time(cube, time_points):
    """Map cube onto new cube with specified time points.

    Missing data inside original bounds is filled with nearest neighbour
    Missing data outside original bounds is masked.
    """
    time_points = cube.coord('time').units.num2date(time_points)
    sample_points = [('time', time_points)]
    scheme = iris.analysis.Nearest(extrapolation_mode='mask')

    # Make sure that all integer time coordinates ('year', 'month',
    # 'day_of_year', etc.) are converted to floats, otherwise the
    # nearest-neighbor interpolation will fail with a "cannot convert float NaN
    # to integer". In addition, remove their bounds (this would be done by iris
    # anyway).
    int_time_coords = []
    for coord in cube.coords(dimensions=cube.coord_dims('time'),
                             dim_coords=False):
        if np.issubdtype(coord.points.dtype, np.integer):
            int_time_coords.append(coord.name())
            coord.points = coord.points.astype(float)
            coord.bounds = None

    # Do the actual interpolation
    try:
        new_cube = cube.interpolate(sample_points, scheme)
    except Exception as excinfo:
        raise ValueError(
            f"Tried to align cubes in multi-model statistics, but failed for "
            f"cube {cube}\n and time points {time_points}") from excinfo

    # Change the dtype of int_time_coords to their original values
    for coord_name in int_time_coords:
        coord = new_cube.coord(coord_name,
                               dimensions=new_cube.coord_dims('time'),
                               dim_coords=False)
        coord.points = coord.points.astype(int)

    return new_cube


def _align(cubes, span):
    """Expand or subset cubes so they share a common time span."""
    _unify_time_coordinates(cubes)

    if _time_coords_are_aligned(cubes):
        return cubes

    all_time_arrays = [cube.coord('time').points for cube in cubes]

    if span == 'overlap':
        new_time_points = reduce(np.intersect1d, all_time_arrays)
    elif span == 'full':
        new_time_points = reduce(np.union1d, all_time_arrays)
    else:
        raise ValueError(f"Invalid argument for span: {span!r}"
                         "Must be one of 'overlap', 'full'.")

    new_cubes = [_map_to_new_time(cube, new_time_points) for cube in cubes]

    for cube in new_cubes:
        # Make sure bounds exist and are consistent
        cube.coord('time').bounds = None
        cube.coord('time').guess_bounds()

    return new_cubes


def _equalise_cell_methods(cubes):
    """Equalise cell methods in cubes (in-place)."""
    # Simply remove all cell methods
    for cube in cubes:
        cube.cell_methods = None


def _equalise_coordinates(cubes):
    """Equalise coordinates in cubes (in-place)."""
    if not cubes:
        return

    # If metadata of a coordinate metadata is equal for all cubes, do not
    # modify it; else remove long_name and attributes.
    equal_coords_metadata = []
    for coord in cubes[0].coords():
        for other_cube in cubes[1:]:
            other_cube_has_equal_coord = [
                coord.metadata == other_coord.metadata for other_coord in
                other_cube.coords(coord.name())
            ]
            if not any(other_cube_has_equal_coord):
                break
        else:
            equal_coords_metadata.append(coord.metadata)

    # Modify coordinates accordingly
    for cube in cubes:
        for coord in cube.coords():
            if coord.metadata not in equal_coords_metadata:
                coord.long_name = None
                coord.attributes = None

        # Additionally remove specific scalar coordinates which are not
        # expected to be equal in the input cubes
        scalar_coords_to_remove = ['p0', 'ptop']
        for scalar_coord in cube.coords(dimensions=()):
            if scalar_coord.var_name in scalar_coords_to_remove:
                cube.remove_coord(scalar_coord)


def _equalise_fx_variables(cubes):
    """Equalise fx variables in cubes (in-place)."""
    # Simple remove all fx variables
    for cube in cubes:
        remove_fx_variables(cube)


def _combine(cubes):
    """Merge iris cubes into a single big cube with new dimension.

    This assumes that all input cubes have the same shape.
    """
    # Equalise some metadata that can cause merge to fail (in-place)
    # https://scitools-iris.readthedocs.io/en/stable/userguide/
    #    merge_and_concat.html#common-issues-with-merge-and-concatenate
    equalise_attributes(cubes)
    _equalise_cell_methods(cubes)
    _equalise_coordinates(cubes)
    _equalise_fx_variables(cubes)

    for i, cube in enumerate(cubes):
        concat_dim = iris.coords.AuxCoord(i, var_name=CONCAT_DIM)
        cube.add_aux_coord(concat_dim)

    cubes = iris.cube.CubeList(cubes)

    merged_cube = cubes.merge_cube()

    return merged_cube


def _compute_slices(cubes):
    """Create cube slices resulting in a combined cube of about 1 GiB."""
    gibibyte = 2**30
    total_bytes = cubes[0].data.nbytes * len(cubes)
    n_slices = int(np.ceil(total_bytes / gibibyte))

    n_timesteps = cubes[0].shape[0]
    slice_len = int(np.ceil(n_timesteps / n_slices))

    for i in range(n_slices):
        start = i * slice_len
        end = (i + 1) * slice_len
        if end >= n_timesteps:
            yield slice(start, n_timesteps)
            return
        yield slice(start, end)


def _compute_eager(cubes: list, *, operator: iris.analysis.Aggregator,
                   **kwargs):
    """Compute statistics one slice at a time."""
    _ = [cube.data for cube in cubes]  # make sure the cubes' data are realized

    result_slices = []
    for chunk in _compute_slices(cubes):
        single_model_slices = [cube[chunk] for cube in cubes]
        combined_slice = _combine(single_model_slices)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=(
                    "Collapsing a non-contiguous coordinate. "
                    f"Metadata may not be fully descriptive for '{CONCAT_DIM}."
                ),
                category=UserWarning,
                module='iris',
            )
            collapsed_slice = combined_slice.collapsed(CONCAT_DIM, operator,
                                                       **kwargs)

        # some iris aggregators modify dtype, see e.g.
        # https://numpy.org/doc/stable/reference/generated/numpy.ma.average.html
        collapsed_slice.data = collapsed_slice.data.astype(np.float32)

        result_slices.append(collapsed_slice)

    try:
        result_cube = iris.cube.CubeList(result_slices).concatenate_cube()
    except Exception as excinfo:
        raise ValueError(
            "Multi-model statistics failed to concatenate results into a"
            f" single array. This happened for operator {operator}"
            f" with computed statistics {result_slices}."
            "This can happen e.g. if the calculation results in inconsistent"
            f" dtypes. Encountered the following exception: {excinfo}")

    result_cube.data = np.ma.array(result_cube.data)
    result_cube.remove_coord(CONCAT_DIM)
    if result_cube.cell_methods:
        cell_method = result_cube.cell_methods[0]
        result_cube.cell_methods = None
        updated_method = iris.coords.CellMethod(
            method=cell_method.method,
            coords=cell_method.coord_names,
            intervals=cell_method.intervals,
            comments=f'input_cubes: {len(cubes)}')
        result_cube.add_cell_method(updated_method)
    return result_cube


def _multicube_statistics(cubes, statistics, span):
    """Compute statistics over multiple cubes.

    Can be used e.g. for ensemble or multi-model statistics.

    Cubes are merged and subsequently collapsed along a new auxiliary
    coordinate. Inconsistent attributes will be removed.
    """
    if len(cubes) == 1:
        raise ValueError('Cannot perform multicube statistics '
                         'for a single cube.')

    copied_cubes = [cube.copy() for cube in cubes]  # avoid modifying inputs
    aligned_cubes = _align(copied_cubes, span=span)

    statistics_cubes = {}
    for statistic in statistics:
        logger.debug('Multicube statistics: computing: %s', statistic)
        operator, kwargs = _resolve_operator(statistic)

        result_cube = _compute_eager(aligned_cubes,
                                     operator=operator,
                                     **kwargs)
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
                           groupby=None,
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

    Parameters
    ----------
    products: list
        Cubes (or products) over which the statistics will be computed.
    span: str
        Overlap or full; if overlap, statitstics are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data.
    statistics: list
        Statistical metrics to be computed, e.g. [``mean``, ``max``]. Choose
        from the operators listed in the iris.analysis package. Percentiles can
        be specified like ``pXX.YY``.
    output_products: dict
        For internal use only. A dict with statistics names as keys and
        preprocessorfiles as values. If products are passed as input, the
        statistics cubes will be assigned to these output products.
    groupby:  tuple
        Group products by a given tag or attribute, e.g.
        ('project', 'dataset', 'tag1').
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
        statistics_products = set()
        for group, input_prods in _group_products(products, by_key=groupby):
            sub_output_products = output_products[group]

            # Compute statistics on a single group
            group_statistics = _multiproduct_statistics(
                products=input_prods,
                statistics=statistics,
                output_products=sub_output_products,
                span=span,
                keep_input_datasets=keep_input_datasets
            )

            statistics_products |= group_statistics

        return statistics_products
    raise ValueError(
        "Input type for multi_model_statistics not understood. Expected "
        "iris.cube.Cube or esmvalcore.preprocessor.PreprocessorFile, "
        "got {}".format(products))


def ensemble_statistics(products, statistics,
                        output_products, span='overlap'):
    """Entry point for ensemble statistics.

    An ensemble grouping is performed on the input products.
    The statistics are then computed calling
    the :func:`esmvalcore.preprocessor.multi_model_statistics` module,
    taking the grouped products as an input.

    Parameters
    ----------
    products: list
        Cubes (or products) over which the statistics will be computed.
    statistics: list
        Statistical metrics to be computed, e.g. [``mean``, ``max``]. Choose
        from the operators listed in the iris.analysis package. Percentiles can
        be specified like ``pXX.YY``.
    output_products: dict
        For internal use only. A dict with statistics names as keys and
        preprocessorfiles as values. If products are passed as input, the
        statistics cubes will be assigned to these output products.
    span: str (default: 'overlap')
        Overlap or full; if overlap, statitstics are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data.

    Returns
    -------
    set
        A set of output_products with the resulting ensemble statistics.

    See Also
    --------
    :func:`esmvalcore.preprocessor.multi_model_statistics` for
    the full description of the core statistics function.
    """
    ensemble_grouping = ('project', 'dataset', 'exp', 'sub_experiment')
    return multi_model_statistics(
        products=products,
        span=span,
        statistics=statistics,
        output_products=output_products,
        groupby=ensemble_grouping,
        keep_input_datasets=False
    )
