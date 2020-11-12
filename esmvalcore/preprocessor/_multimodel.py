"""Statistics across cubes.

This module contains functions to compute statistics across multiple cubes or
products.

Ensemble statistics uses iris built in functions and support lazy evaluation.
Multi-model statistics uses custom functions that operate directly on numpy
arrays.

Wrapper functions separate esmvalcore internals, operating on products, from
generalized functions that operate on iris cubes. These wrappers support
grouped execution by passing a groupby keyword.
"""
import copy
import logging
import re
from datetime import datetime
from functools import partial, reduce

import cf_units
import iris
import numpy as np
import scipy
from iris.experimental.equalise_cubes import equalise_attributes

from ._other import _group_products

logger = logging.getLogger(__name__)


def _plev_fix(dataset, pl_idx):
    """Extract valid plev data.

    this function takes care of situations in which certain plevs are
    completely masked due to unavailable interpolation boundaries.
    """
    if np.ma.is_masked(dataset):
        # keep only the valid plevs
        if not np.all(dataset.mask[pl_idx]):
            statj = np.ma.array(dataset[pl_idx], mask=dataset.mask[pl_idx])
        else:
            logger.debug('All vals in plev are masked, ignoring.')
            statj = None
    else:
        mask = np.zeros_like(dataset[pl_idx], bool)
        statj = np.ma.array(dataset[pl_idx], mask=mask)

    return statj


def _quantile(data, axis, quantile):
    """Calculate quantile.

    Workaround for calling scipy's mquantiles with arrays of >2
    dimensions Similar to iris' _percentiles function, see their
    discussion: https://github.com/SciTools/iris/pull/625
    """
    # Ensure that the target axis is the last dimension.
    data = np.rollaxis(data, axis, start=data.ndim)
    shape = data.shape[:-1]
    # Flatten any leading dimensions.
    if shape:
        data = data.reshape([np.prod(shape), data.shape[-1]])
    # Perform the quantile calculation.
    result = scipy.stats.mstats.mquantiles(data,
                                           quantile,
                                           axis=-1,
                                           alphap=1,
                                           betap=1)
    # Ensure to unflatten any leading dimensions.
    if shape:
        result = result.reshape(shape)
    # Check whether to reduce to a scalar result
    if result.shape == (1, ):
        result = result[0]

    return result


def _compute_statistic(data, statistic_name):
    """Compute multimodel statistic."""
    data = np.ma.array(data)
    statistic = data[0]

    if statistic_name == 'median':
        statistic_function = np.ma.median
    elif statistic_name == 'mean':
        statistic_function = np.ma.mean
    elif statistic_name == 'std':
        statistic_function = np.ma.std
    elif statistic_name == 'max':
        statistic_function = np.ma.max
    elif statistic_name == 'min':
        statistic_function = np.ma.min
    elif re.match(r"^(p\d{1,2})(\.\d*)?$", statistic_name):
        # percentiles between p0 and p99.99999...
        quantile = float(statistic_name[1:]) / 100
        statistic_function = partial(_quantile, quantile=quantile)
    else:
        raise ValueError(f'No such statistic: `{statistic_name}`')

    # no plevs
    if len(data[0].shape) < 3:
        # get all NOT fully masked data - u_data
        # data is per time point
        # so we can safely NOT compute stats for single points
        if data.ndim == 1:
            u_datas = data
        else:
            u_datas = [d for d in data if not np.all(d.mask)]
        if len(u_datas) > 1:
            statistic = statistic_function(data, axis=0)
        else:
            statistic.mask = True
        return statistic

    # plevs
    for j in range(statistic.shape[0]):
        plev_check = []
        for cdata in data:
            fixed_data = _plev_fix(cdata, j)
            if fixed_data is not None:
                plev_check.append(fixed_data)

        # check for nr datasets
        if len(plev_check) > 1:
            plev_check = np.ma.array(plev_check)
            statistic[j] = statistic_function(plev_check, axis=0)
        else:
            statistic.mask[j] = True

    return statistic


def _put_in_cube(template_cube, cube_data, statistic, t_axis):
    """Quick cube building and saving."""
    tunits = template_cube.coord('time').units
    times = iris.coords.DimCoord(t_axis, standard_name='time', units=tunits)
    times.bounds = None
    times.guess_bounds()

    coord_names = [c.long_name for c in template_cube.coords()]
    coord_names.extend([c.standard_name for c in template_cube.coords()])
    if 'latitude' in coord_names:
        lats = template_cube.coord('latitude')
    else:
        lats = None
    if 'longitude' in coord_names:
        lons = template_cube.coord('longitude')
    else:
        lons = None

    # no plevs
    if len(template_cube.shape) == 3:
        cspec = [(times, 0), (lats, 1), (lons, 2)]
    # plevs
    elif len(template_cube.shape) == 4:
        plev = template_cube.coord('air_pressure')
        cspec = [(times, 0), (plev, 1), (lats, 2), (lons, 3)]
    elif len(template_cube.shape) == 1:
        cspec = [
            (times, 0),
        ]
    elif len(template_cube.shape) == 2:
        # If you're going to hardwire air_pressure into this,
        # might as well have depth here too.
        plev = template_cube.coord('depth')
        cspec = [
            (times, 0),
            (plev, 1),
        ]

    # correct dspec if necessary
    fixed_dspec = np.ma.fix_invalid(cube_data, copy=False, fill_value=1e+20)
    # put in cube
    stats_cube = iris.cube.Cube(fixed_dspec,
                                dim_coords_and_dims=cspec,
                                long_name=statistic)
    coord_names = [coord.name() for coord in template_cube.coords()]
    if 'air_pressure' in coord_names:
        if len(template_cube.shape) == 3:
            stats_cube.add_aux_coord(template_cube.coord('air_pressure'))

    stats_cube.var_name = template_cube.var_name
    stats_cube.long_name = template_cube.long_name
    stats_cube.standard_name = template_cube.standard_name
    stats_cube.units = template_cube.units
    return stats_cube


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
    1st of july for yearly data (consistent with `regrid_time`), so that there
    are no mismatches in the time arrays.

    If cubes have different time units, it will use reset the calendar to a
    default gregorian calendar with unit "days since 1850-01-01".

    Might not work for (sub)daily data, because different calendars may have
    different number of days in the year.
    """
    t_unit = _get_consistent_time_unit(cubes)

    for cube in cubes:
        # Extract date info from cube
        coord = cube.coord('time')
        years = [p.year for p in coord.units.num2date(coord.points)]
        months = [p.month for p in coord.units.num2date(coord.points)]

        # Reconstruct default calendar
        if 0 not in np.diff(years):
            # yearly data
            dates = [datetime(year, 7, 1) for year in years]

        elif 0 not in np.diff(months):
            # monthly data
            dates = [
                datetime(year, month, 15)
                for year, month in zip(years, months)
            ]
        else:
            # (sub)daily data
            coord = cube.coord('time')
            if coord.units != t_unit:
                logger.warning(
                    "Multimodel encountered (sub)daily data and inconsistent "
                    "time units or calendars. Attempting to continue, but "
                    "might produce unexpected results.")
            dates = coord.units.num2date(coord.points)

        # Update the cubes' time coordinate (both point values and the units!)
        cube.coord('time').points = t_unit.date2num(dates)
        cube.coord('time').units = t_unit
        cube.coord('time').bounds = None
        cube.coord('time').guess_bounds()


def _get_time_slice(cubes, time):
    """Fill time slice array with cubes' data if time in cube, else mask."""
    time_slice = []
    for cube in cubes:
        cube_time = cube.coord('time').points
        if time in cube_time:
            idx = int(np.argwhere(cube_time == time))
            subset = cube.data[idx]
        else:
            subset = np.ma.empty(list(cube.shape[1:]))
            subset.mask = True
        time_slice.append(subset)
    return time_slice


def _assemble_data(cubes, statistic, span='overlap'):
    """Get statistical data in iris cubes."""
    # New time array representing the union or intersection of all cubes
    time_spans = [cube.coord('time').points for cube in cubes]
    if span == 'overlap':
        new_times = reduce(np.intersect1d, time_spans)
    elif span == 'full':
        new_times = reduce(np.union1d, time_spans)
    n_times = len(new_times)

    # Target array to populate with computed statistics
    new_shape = [n_times] + list(cubes[0].shape[1:])
    stats_data = np.ma.zeros(new_shape, dtype=np.dtype('float32'))

    # Realize all cubes at once instead of separately for each time slice
    _ = [cube.data for cube in cubes]

    # Make time slices and compute stats
    for i, time in enumerate(new_times):
        time_data = _get_time_slice(cubes, time)
        stats_data[i] = _compute_statistic(time_data, statistic)

    template = cubes[0]
    stats_cube = _put_in_cube(template, stats_data, statistic, new_times)
    return stats_cube


def multicube_statistics(cubes, statistics, span):
    """Compute statistics across input cubes.

    This function deals with non-homogeneous cubes by taking the time union
    (span='full') or intersection (span='overlap'), and extending or subsetting
    the cubes as necessary. Apart from the time coordinate, cubes must have
    consistent shapes.

    This function operates directly on numpy (masked) arrays and rebuilds the
    resulting cubes from scratch. Therefore, it is not suitable for lazy
    evaluation.

    This function is restricted to maximum four-dimensional data: time,
    vertical axis, two horizontal axes.

    Parameters
    ----------
    cubes: list
        list of cubes to be used in multimodel stat computation;
    statistics: list
        statistical measures to be computed. Available options: mean, median,
        max, min, std, or pXX.YY (for percentile XX.YY; decimal part optional).
    span: str
        overlap or full; if overlap, statitsticss are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data.

    Returns
    -------
    dict
        dictionary of statistics cubes with statistics' names as keys.

    Raises
    ------
    ValueError
        If span is neither overlap nor full.
    """
    logger.debug('Multimodel statistics: computing: %s', statistics)

    # Reset time coordinates and make cubes share the same calendar
    _unify_time_coordinates(cubes)

    if span == 'overlap':
        # check if we have any time overlap
        times = [cube.coord('time').points for cube in cubes]
        overlap = reduce(np.intersect1d, times)
        if len(overlap) <= 1:
            logger.info("Time overlap between cubes is none or a single point."
                        "check datasets: will not compute statistics.")
            return cubes
        logger.debug("Using common time overlap between "
                     "datasets to compute statistics.")
    elif span == 'full':
        logger.debug("Using full time spans to compute statistics.")
    else:
        raise ValueError(
            "Unexpected value for span {}, choose from 'overlap', 'full'".
            format(span))

    statistics_cubes = {}
    for statistic in statistics:
        # Compute statistic
        statistic_cube = _assemble_data(cubes, statistic, span)
        statistics_cubes[statistic] = statistic_cube

    return statistics_cubes


def multicube_statistics_iris(cubes, statistics):
    """Compute statistics across input cubes.

    Like multicube_statistics, but operates directly on Iris cubes.

    Cubes are merged and subsequently collapsed along a new auxiliary
    coordinate. Inconsistent attributes will be removed.

    This method uses iris' built in functions, exposing the operators in
    iris.analysis and supporting lazy evaluation. Input cubes must have
    consistent shapes.

    Note: some of the operators in iris.analysis require additional arguments,
    such as percentiles or weights. These operators are currently not
    supported.

    Parameters
    ----------
    cubes: list
        list of cubes to be used in multimodel stat computation;
    statistics: list
        statistical measures to be computed. Choose from the operators
        listed in the iris.analysis package.

    Returns
    -------
    dict
        dictionary of statistics cubes with statistics' names as keys.
    """
    operators = vars(iris.analysis)

    cubes = copy.deepcopy(cubes)

    for i, cube in enumerate(cubes):
        concat_dim = iris.coords.AuxCoord(i, var_name='ens')
        cube.add_aux_coord(concat_dim)

    equalise_attributes(cubes)

    cubes = iris.cube.CubeList(cubes)
    cube = cubes.merge_cube()

    statistics_cubes = {}
    for statistic in statistics:
        try:
            operator = operators[statistic.upper()]
        except KeyError as err:
            raise ValueError(
                f'Statistic `{statistic}` not supported in',
                '`ensemble_statistics`. Choose supported operator from',
                '`iris.analysis package`.') from err

        # this will always return a masked array
        statistic_cube = cube.collapsed('ens', operator)
        statistics_cubes[statistic] = statistic_cube

    return statistics_cubes


def _multiproduct_statistics(products,
                             statistics,
                             output_products,
                             span='overlap',
                             engine='esmvalcore'):
    """Compute statistics across products.

    Extract cubes from products, calculate the statistics across cubes and
    assign the resulting output cubes to the output_products.

    Parameters
    ----------
    products: list
        list of PreprocessorFile's
    statistics: list
        list of strings describing the statistics that will be computed
    output_products: dict
        dict of PreprocessorFile's with statistic names as keys.
    span: str
        'full' or 'overlap', whether to calculate the statistics on the
        time union ('full') or time intersection ('overlap') of the cubes.
    engine: str
        'iris' or 'esmvalcore', which function to use to compute the
        statistics. Iris is more efficient and exposes more operators,
        but requires highly homogeneous/compatible cubes.

    Returns
    -------
    set
        set of PreprocessorFiles containing the computed statistics cubes.
    """
    if engine == 'iris':
        aggregator = multicube_statistics_iris
    else:
        aggregator = partial(multicube_statistics, span=span)

    # Extract cubes from products
    cubes = [cube for product in products for cube in product.cubes]

    # Compute statistics
    if len(cubes) < 2:
        logger.info('Found only 1 cube; no statistics computed for %r',
                    list(products)[0])
        statistics_cubes = {statistic: cubes[0] for statistic in statistics}
    else:
        statistics_cubes = aggregator(cubes=cubes, statistics=statistics)

    # Add cubes to output products and log provenance
    statistics_products = set()
    for statistic, cube in statistics_cubes.items():
        statistics_product = output_products[statistic]
        statistics_product.cubes = [cube]

        for product in products:
            statistics_product.wasderivedfrom(product)

        logger.info("Generated %s", statistics_product)
        statistics_products.add(statistics_product)

    return statistics_products


def multi_model_statistics(input_products: set,
                           statistics: list,
                           output_products: set,
                           groupby: str = None,
                           span: str = 'overlap',
                           engine: str = 'esmvalcore'):
    """Entry point for multi model statistics.

    The products are grouped (if groupby argument is specified) and the
    cubes are extracted from their products. Resulting cubes are added to
    their corresponding `output_products`.

    See Also
    --------
    multicube_statistics : core statistics function.
    """
    statistics_products = set()
    for identifier, products in _group_products(input_products, by=groupby):
        sub_output_products = output_products[identifier]

        # Compute statistics on a single group
        group_statistics = _multiproduct_statistics(
            products=products,
            statistics=statistics,
            output_products=sub_output_products,
            span=span,
            engine=engine,
        )

        statistics_products |= group_statistics

    return statistics_products


def ensemble_statistics(input_products: set, statistics: list,
                        output_products: set):
    """Entry point for ensemble statistics.

    The products are grouped and the cubes are extracted from
    the products. Resulting cubes are assigned to `output_products`.

    See Also
    --------
    multicube_statistics_iris : core statistics function.
    """
    ensemble_grouping = ('project', 'dataset', 'exp')
    return multi_model_statistics(
        input_products=input_products,
        statistics=statistics,
        output_products=output_products,
        groupby=ensemble_grouping,
        engine='iris',
    )
