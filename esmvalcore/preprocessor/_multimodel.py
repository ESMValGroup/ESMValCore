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
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import MergeError
from iris.util import equalise_attributes, new_axis

from esmvalcore.iris_helpers import date2num
from esmvalcore.preprocessor._supplementary_vars import (
    remove_supplementary_variables,
)

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
        _guess_time_bounds(cube)


def _guess_time_bounds(cube):
    """Guess time bounds if possible."""
    cube.coord('time').bounds = None
    if cube.coord('time').shape == (1,):
        logger.debug(
            "Encountered scalar time coordinate in multi_model_statistics: "
            "cannot determine its bounds"
        )
    else:
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
    time_coord = cube.coord('time')

    # Try if the required time points can be obtained by slicing the cube.
    time_slice = np.isin(time_coord.points, time_points)
    if np.any(time_slice) and np.array_equal(time_coord.points[time_slice],
                                             time_points):
        time_idx, = cube.coord_dims('time')
        indices = tuple(time_slice if i == time_idx else slice(None)
                        for i in range(cube.ndim))
        return cube[indices]

    time_points = time_coord.units.num2date(time_points)
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
        additional_info = ""
        if cube.coords('time', dimensions=()):
            additional_info = (
                " Note: this alignment does not work for scalar time "
                "coordinates. To ignore all scalar coordinates in the input "
                "data, use the preprocessor option "
                "`ignore_scalar_coords=True`."
            )
        raise ValueError(
            f"Tried to align cubes in multi-model statistics, but failed for "
            f"cube {cube}\n and time points {time_points}.{additional_info}"
        ) from excinfo

    # Change the dtype of int_time_coords to their original values
    for coord_name in int_time_coords:
        coord = new_cube.coord(coord_name,
                               dimensions=new_cube.coord_dims('time'),
                               dim_coords=False)
        coord.points = coord.points.astype(int)

    return new_cube


def _align_time_coord(cubes, span):
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
        _guess_time_bounds(cube)

    return new_cubes


def _equalise_cell_methods(cubes):
    """Equalise cell methods in cubes (in-place)."""
    # Simply remove all cell methods
    for cube in cubes:
        cube.cell_methods = None


def _get_equal_coords_metadata(cubes):
    """Get metadata for exactly matching coordinates across cubes."""
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
    return equal_coords_metadata


def _get_equal_coord_names_metadata(cubes, equal_coords_metadata):
    """Get metadata for coords with matching names and units across cubes.

    Note
    ----
    Ignore coordinates whose names are not unique.

    """
    equal_names_metadata = {}
    for coord in cubes[0].coords():
        coord_name = coord.name()

        # Ignore exactly matching coordinates
        if coord.metadata in equal_coords_metadata:
            continue

        # Ignore coordinates that are not unique in original cube
        if len(cubes[0].coords(coord_name)) > 1:
            continue

        # Check if coordinate names and units match across all cubes
        for other_cube in cubes[1:]:

            # Ignore names that do not exist in other cube/are not unique
            if len(other_cube.coords(coord_name)) != 1:
                break

            # Ignore names where units do not match across cubes
            if coord.units != other_cube.coord(coord_name).units:
                break

        # Coordinate name exists in all other cubes with identical units
        # --> Get metadata that is identical across all cubes
        else:
            std_names = list(
                {c.coord(coord_name).standard_name for c in cubes}
            )
            long_names = list(
                {c.coord(coord_name).long_name for c in cubes}
            )
            var_names = list(
                {c.coord(coord_name).var_name for c in cubes}
            )
            equal_names_metadata[coord_name] = dict(
                standard_name=std_names[0] if len(std_names) == 1 else None,
                long_name=long_names[0] if len(long_names) == 1 else None,
                var_name=var_names[0] if len(var_names) == 1 else None,
            )

    return equal_names_metadata


def _equalise_coordinate_metadata(cubes):
    """Equalise coordinates in cubes (in-place)."""
    if not cubes:
        return

    # Filter out coordinates with exactly matching metadata across all cubes
    # --> these will not be modified at all
    equal_coords_metadata = _get_equal_coords_metadata(cubes)

    # Filter out coordinates with matching names and units
    # --> keep matching names of these coordinates
    # Note: ignores duplicate coordinates
    equal_names_metadata = _get_equal_coord_names_metadata(
        cubes,
        equal_coords_metadata
    )

    # Modify all coordinates of all cubes accordingly
    for cube in cubes:
        for coord in cube.coords():

            # Exactly matching coordinates --> do not modify
            if coord.metadata in equal_coords_metadata:
                continue

            # Non-exactly matching coordinates --> first, delete attributes and
            # circular property
            coord.attributes = {}
            if isinstance(coord, DimCoord):
                coord.circular = False

            # Matching names and units --> set common names
            if coord.name() in equal_names_metadata:
                equal_names = equal_names_metadata[coord.name()]
                coord.standard_name = equal_names['standard_name']
                coord.long_name = equal_names['long_name']
                coord.var_name = equal_names['var_name']
                continue

            # Remaining coordinates --> remove long_name
            # Note: remaining differences will raise an error at a later stage
            coord.long_name = None

        # Remove special scalar coordinates which are not expected to be equal
        # in the input cubes. Note: if `ignore_scalar_coords=True` is used for
        # `multi_model_statistics`, the cubes do not contain scalar coordinates
        # at this point anymore.
        scalar_coords_to_always_remove = ['p0', 'ptop']
        for scalar_coord in cube.coords(dimensions=()):
            if scalar_coord.var_name in scalar_coords_to_always_remove:
                cube.remove_coord(scalar_coord)
                logger.debug(
                    "Removed scalar coordinate '%s' from cube %s",
                    scalar_coord.var_name,
                    cube.summary(shorten=True),
                )


def _equalise_fx_variables(cubes):
    """Equalise fx variables in cubes (in-place)."""
    # Simple remove all fx variables
    for cube in cubes:
        remove_supplementary_variables(cube)


def _equalise_var_metadata(cubes):
    """Equalise variable metadata in cubes (in-place).

    If cubes have the same ``name()`` and ``units``, assign identical
    `standard_names`, `long_names`, and `var_names`.

    """
    attrs = ['standard_name', 'long_name', 'var_name']
    equal_names_metadata = {}

    # Collect all names from the different cubes, grouped by cube.name() and
    # cube.units (ignore `None`)
    for cube in cubes:
        cube_id = f"{cube.name()} ({cube.units})"
        equal_names_metadata.setdefault(cube_id, {a: set() for a in attrs})
        for attr in attrs:
            val = getattr(cube, attr)
            if val is not None:
                equal_names_metadata[cube_id][attr].add(val)

    # Unify names (always use first encountered value, even if there are
    # different values)
    for names in equal_names_metadata.values():
        for attr in attrs:
            vals = sorted(names[attr])
            if not vals:  # all names were `None`
                names[attr] = None
            else:  # always use first encountered value
                names[attr] = vals[0]

    # Assign equal names for cubes with identical cube.name() and cube.units
    for cube in cubes:
        cube_id = f"{cube.name()} ({cube.units})"
        for attr in attrs:
            setattr(cube, attr, equal_names_metadata[cube_id][attr])


def _combine(cubes):
    """Merge iris cubes into a single big cube with new dimension.

    This assumes that all input cubes have the same shape.
    """
    # Equalise some metadata that can cause merge to fail (in-place)
    # https://scitools-iris.readthedocs.io/en/stable/userguide/
    #    merge_and_concat.html#common-issues-with-merge-and-concatenate
    equalise_attributes(cubes)
    _equalise_var_metadata(cubes)
    _equalise_cell_methods(cubes)
    _equalise_coordinate_metadata(cubes)
    _equalise_fx_variables(cubes)

    for i, cube in enumerate(cubes):
        concat_dim = iris.coords.AuxCoord(i, var_name=CONCAT_DIM)
        cube.add_aux_coord(concat_dim)

    cubes = CubeList(cubes)

    # For a single cube, merging returns a scalar CONCAT_DIM, which leads to a
    # "Cannot collapse a dimension which does not describe any data" error when
    # collapsing. Thus, treat single cubes differently here.
    if len(cubes) == 1:
        return new_axis(cubes[0], scalar_coord=CONCAT_DIM)

    try:
        merged_cube = cubes.merge_cube()
    except MergeError as exc:
        # Note: str(exc) starts with "failed to merge into a single cube.\n"
        # --> remove this here for clear error message
        msg = "\n".join(str(exc).split('\n')[1:])
        raise ValueError(
            f"Multi-model statistics failed to merge input cubes into a "
            f"single array:\n{cubes}\n{msg}"
        ) from exc

    return merged_cube


def _compute_slices(cubes):
    """Create cube slices along the first dimension of the cubes.

    This results in a combined cube of about 1 GiB.

    Note
    ----
    For scalar cubes, simply return ``None``.

    """
    # Scalar cubes
    if cubes[0].shape == ():
        yield None
        return

    # Non-scalar cubes
    gibibyte = 2**30
    total_bytes = cubes[0].data.nbytes * len(cubes)
    n_slices = int(np.ceil(total_bytes / gibibyte))

    len_dim_0 = cubes[0].shape[0]
    slice_len = int(np.ceil(len_dim_0 / n_slices))

    for i in range(n_slices):
        start = i * slice_len
        end = (i + 1) * slice_len
        if end >= len_dim_0:
            yield slice(start, len_dim_0)
            return
        yield slice(start, end)


def _compute_eager(
    cubes: list,
    *,
    operator: iris.analysis.Aggregator,
    **kwargs,
):
    """Compute statistics one slice at a time."""
    _ = [cube.data for cube in cubes]  # make sure the cubes' data are realized

    result_slices = iris.cube.CubeList()
    for chunk in _compute_slices(cubes):
        if chunk is None:
            input_slices = cubes  # scalar cubes
        else:
            input_slices = [cube[chunk] for cube in cubes]
        result_slice = _compute(input_slices, operator=operator, **kwargs)
        result_slices.append(result_slice)

    try:
        result_cube = result_slices.concatenate_cube()
    except Exception as excinfo:
        raise ValueError(
            f"Multi-model statistics failed to concatenate results into a "
            f"single array. This happened for operator {operator} "
            f"with computed statistics {result_slices}. "
            f"This can happen e.g. if the calculation results in inconsistent "
            f"dtypes") from excinfo

    result_cube.data = np.ma.array(result_cube.data)

    return result_cube


def _compute(cubes: list, *, operator: iris.analysis.Aggregator, **kwargs):
    """Compute statistic."""
    cube = _combine(cubes)

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
        warnings.filterwarnings(
            'ignore',
            message=(
                f"Cannot check if coordinate is contiguous: Invalid "
                f"operation for '{CONCAT_DIM}'"
            ),
            category=UserWarning,
            module='iris',
        )
        # This will always return a masked array
        result_cube = cube.collapsed(CONCAT_DIM, operator, **kwargs)

    # Remove concatenation dimension added by _combine
    result_cube.remove_coord(CONCAT_DIM)
    for cube in cubes:
        cube.remove_coord(CONCAT_DIM)

    # some iris aggregators modify dtype, see e.g.
    # https://numpy.org/doc/stable/reference/generated/numpy.ma.average.html
    result_cube.data = result_cube.core_data().astype(np.float32)

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


def _multicube_statistics(cubes, statistics, span, ignore_scalar_coords=False):
    """Compute statistics over multiple cubes.

    Can be used e.g. for ensemble or multi-model statistics.

    Cubes are merged and subsequently collapsed along a new auxiliary
    coordinate. Inconsistent attributes will be removed.
    """
    if not cubes:
        raise ValueError(
            "Cannot perform multicube statistics for an empty list of cubes"
        )

    # Avoid modifying inputs
    cubes = [cube.copy() for cube in cubes]

    # Remove scalar coordinates in input cubes if desired to ignore them when
    # merging
    if ignore_scalar_coords:
        for cube in cubes:
            for scalar_coord in cube.coords(dimensions=()):
                cube.remove_coord(scalar_coord)
                logger.debug(
                    "Removed scalar coordinate '%s' from cube %s since "
                    "ignore_scalar_coords=True",
                    scalar_coord.var_name,
                    cube.summary(shorten=True),
                )

    # If all cubes contain a time coordinate, align them. If no cube contains a
    # time coordinate, do nothing. Else, raise an exception.
    time_coords = [cube.coords('time') for cube in cubes]
    if all(time_coords):
        cubes = _align_time_coord(cubes, span=span)
    elif not any(time_coords):
        pass
    else:
        raise ValueError(
            "Multi-model statistics failed to merge input cubes into a single "
            "array: some cubes have a 'time' dimension, some do not have a "
            "'time' dimension."
        )

    # Calculate statistics
    statistics_cubes = {}
    lazy_input = any(cube.has_lazy_data() for cube in cubes)
    for statistic in statistics:
        logger.debug('Multicube statistics: computing: %s', statistic)
        operator, kwargs = _resolve_operator(statistic)
        if lazy_input and operator.lazy_func is not None:
            result_cube = _compute(cubes, operator=operator, **kwargs)
        else:
            result_cube = _compute_eager(cubes, operator=operator, **kwargs)
        statistics_cubes[statistic] = result_cube

    return statistics_cubes


def _multiproduct_statistics(products,
                             statistics,
                             output_products,
                             span=None,
                             keep_input_datasets=None,
                             ignore_scalar_coords=False):
    """Compute multi-cube statistics on ESMValCore products.

    Extract cubes from products, calculate multicube statistics and
    assign the resulting output cubes to the output_products.
    """
    cubes = [cube for product in products for cube in product.cubes]
    statistics_cubes = _multicube_statistics(
        cubes=cubes,
        statistics=statistics,
        span=span,
        ignore_scalar_coords=ignore_scalar_coords,
    )
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
                           keep_input_datasets=True,
                           ignore_scalar_coords=False):
    """Compute multi-model statistics.

    This function computes multi-model statistics on a list of ``products``,
    which can be instances of :py:class:`~iris.cube.Cube` or
    :py:class:`~esmvalcore.preprocessor.PreprocessorFile`.
    The latter is used internally by ESMValCore to store
    workflow and provenance information, and this option should typically be
    ignored.

    Cubes must have consistent shapes apart from a potential time dimension.
    There are two options to combine time coordinates of different lengths, see
    the ``span`` argument.

    Uses the statistical operators in :py:mod:`iris.analysis`, including
    ``mean``, ``median``, ``min``, ``max``, and ``std``. Percentiles are also
    supported and can be specified like ``pXX.YY`` (for percentile ``XX.YY``;
    decimal part optional).

    This function can handle cubes with differing metadata:

    - Cubes with identical :meth:`~iris.coords.Coord.name` and
      :attr:`~iris.coords.Coord.units` will get identical values for
      :attr:`~iris.coords.Coord.standard_name`,
      :attr:`~iris.coords.Coord.long_name`, and
      :attr:`~iris.coords.Coord.var_name` (which will be arbitrarily set to the
      first encountered value if different cubes have different values for
      them).
    - :attr:`~iris.cube.Cube.attributes`: Differing attributes are deleted,
      see :func:`iris.util.equalise_attributes`.
    - :attr:`~iris.cube.Cube.cell_methods`: All cell methods are deleted
      prior to combining cubes.
    - :meth:`~iris.cube.Cube.cell_measures`: All cell measures are deleted
      prior to combining cubes, see
      :func:`esmvalcore.preprocessor.remove_fx_variables`.
    - :meth:`~iris.cube.Cube.ancillary_variables`: All ancillary variables
      are deleted prior to combining cubes, see
      :func:`esmvalcore.preprocessor.remove_fx_variables`.
    - :meth:`~iris.cube.Cube.coords`: Exactly identical coordinates are
      preserved. For coordinates with equal :meth:`~iris.coords.Coord.name` and
      :attr:`~iris.coords.Coord.units`, names are equalized,
      :attr:`~iris.coords.Coord.attributes` deleted and
      :attr:`~iris.coords.DimCoord.circular` is set to ``False``. For all other
      coordinates, :attr:`~iris.coords.Coord.long_name` is removed,
      :attr:`~iris.coords.Coord.attributes` deleted and
      :attr:`~iris.coords.DimCoord.circular` is set to ``False``. Scalar
      coordinates can be removed if desired by the option
      ``ignore_scalar_coords=True``. Please note that some special scalar
      coordinates which are expected to differ across cubes (ancillary
      coordinates for derived coordinates like `p0` and `ptop`) are always
      removed.

    Notes
    -----
    Some of the operators in :py:mod:`iris.analysis` require additional
    arguments. Except for percentiles, these operators are currently not
    supported.

    Lazy operation is supported for all statistics, except ``median``.

    Parameters
    ----------
    products: list
        Cubes (or products) over which the statistics will be computed.
    span: str
        Overlap or full; if overlap, statitstics are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data. This option is ignored if input cubes do not have time
        dimensions.
    statistics: list
        Statistical metrics to be computed, e.g. [``mean``, ``max``]. Choose
        from the operators listed in the iris.analysis package. Percentiles can
        be specified like ``pXX.YY``.
    output_products: dict
        For internal use only. A dict with statistics names as keys and
        preprocessorfiles as values. If products are passed as input, the
        statistics cubes will be assigned to these output products.
    groupby:  tuple
        Group products by a given tag or attribute, e.g., ('project',
        'dataset', 'tag1'). This is ignored if ``products`` is a list of cubes.
    keep_input_datasets: bool
        If True, the output will include the input datasets.
        If False, only the computed statistics will be returned.
    ignore_scalar_coords: bool
        If True, remove any scalar coordinate in the input datasets before
        merging the input cubes into the multi-dataset cube. The resulting
        multi-dataset cube will have no scalar coordinates (the actual input
        datasets will remain unchanged). If False, scalar coordinates will
        remain in the input datasets, which might lead to merge conflicts in
        case the input datasets have different scalar coordinates.

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
    if all(isinstance(p, Cube) for p in products):
        return _multicube_statistics(
            cubes=products,
            statistics=statistics,
            span=span,
            ignore_scalar_coords=ignore_scalar_coords,
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
                keep_input_datasets=keep_input_datasets,
                ignore_scalar_coords=ignore_scalar_coords,
            )

            statistics_products |= group_statistics

        return statistics_products
    raise ValueError(
        f"Input type for multi_model_statistics not understood. Expected "
        f"iris.cube.Cube or esmvalcore.preprocessor.PreprocessorFile, "
        f"got {products}"
    )


def ensemble_statistics(products, statistics,
                        output_products, span='overlap',
                        ignore_scalar_coords=False):
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
    ignore_scalar_coords: bool
        If True, remove any scalar coordinate in the input datasets before
        merging the input cubes into the multi-dataset cube. The resulting
        multi-dataset cube will have no scalar coordinates (the actual input
        datasets will remain unchanged). If False, scalar coordinates will
        remain in the input datasets, which might lead to merge conflicts in
        case the input datasets have different scalar coordinates.

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
        keep_input_datasets=False,
        ignore_scalar_coords=ignore_scalar_coords,
    )
