"""Time operations on cubes.

Allows for selecting data subsets using certain time bounds;
constructing seasonal and area averages.
"""
import copy
import datetime
import logging
from warnings import filterwarnings

import dask.array as da
import iris
import iris.coord_categorisation
import iris.cube
import iris.exceptions
import iris.util
import numpy as np
from iris.time import PartialDateTime

from ._shared import get_iris_analysis_operation, operator_accept_weights

logger = logging.getLogger(__name__)

# Ignore warnings about missing bounds where those are not required
for _coord in (
        'clim_season',
        'day_of_year',
        'day_of_month',
        'month_number',
        'season_year',
        'year',
):
    filterwarnings(
        'ignore',
        "Collapsing a non-contiguous coordinate. "
        "Metadata may not be fully descriptive for '{0}'.".format(_coord),
        category=UserWarning,
        module='iris',
    )


def extract_time(cube, start_year, start_month, start_day, end_year, end_month,
                 end_day):
    """
    Extract a time range from a cube.

    Given a time range passed in as a series of years, months and days, it
    returns a time-extracted cube with data only within the specified
    time range.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    start_year: int
        start year
    start_month: int
        start month
    start_day: int
        start day
    end_year: int
        end year
    end_month: int
        end month
    end_day: int
        end day

    Returns
    -------
    iris.cube.Cube
        Sliced cube.

    Raises
    ------
    ValueError
        if time ranges are outside the cube time limits

    """
    time_coord = cube.coord('time')
    time_units = time_coord.units
    if time_units.calendar == '360_day':
        if start_day > 30:
            start_day = 30
        if end_day > 30:
            end_day = 30
    t_1 = PartialDateTime(
        year=int(start_year), month=int(start_month), day=int(start_day))
    t_2 = PartialDateTime(
        year=int(end_year), month=int(end_month), day=int(end_day))

    constraint = iris.Constraint(
        time=lambda t: t_1 <= t.point < t_2)

    cube_slice = cube.extract(constraint)
    if cube_slice is None:
        raise ValueError(
            f"Time slice {start_year:0>4d}-{start_month:0>2d}-{start_day:0>2d}"
            f" to {end_year:0>4d}-{end_month:0>2d}-{end_day:0>2d} is outside "
            f"cube time bounds {time_coord.cell(0)} to {time_coord.cell(-1)}."
        )

    # Issue when time dimension was removed when only one point as selected.
    if cube_slice.ndim != cube.ndim:
        if cube_slice.coord('time') == time_coord:
            logger.debug('No change needed to time.')
            return cube

    return cube_slice


def extract_season(cube, season):
    """
    Slice cube to get only the data belonging to a specific season.

    Parameters
    ----------
    cube: iris.cube.Cube
        Original data
    season: str
        Season to extract. Available: DJF, MAM, JJA, SON

    Returns
    -------
    iris.cube.Cube
        data cube for specified season.
    """
    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    if not cube.coords('season_year'):
        iris.coord_categorisation.add_season_year(cube,
                                                  'time',
                                                  name='season_year')
    return cube.extract(iris.Constraint(clim_season=season.lower()))


def extract_month(cube, month):
    """
    Slice cube to get only the data belonging to a specific month.

    Parameters
    ----------
    cube: iris.cube.Cube
        Original data
    month: int
        Month to extract as a number from 1 to 12

    Returns
    -------
    iris.cube.Cube
        data cube for specified month.
    """
    if month not in range(1, 13):
        raise ValueError('Please provide a month number between 1 and 12.')
    if not cube.coords('month_number'):
        iris.coord_categorisation.add_month_number(cube, 'time',
                                                   name='month_number')
    return cube.extract(iris.Constraint(month_number=month))


def get_time_weights(cube):
    """
    Compute the weighting of the time axis.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    Returns
    -------
    numpy.array
        Array of time weights for averaging.
    """
    time = cube.coord('time')
    time_weights = time.bounds[..., 1] - time.bounds[..., 0]
    time_weights = time_weights.squeeze()
    if time_weights.shape == ():
        time_weights = da.broadcast_to(time_weights, cube.shape)
    else:
        time_weights = iris.util.broadcast_to_shape(time_weights, cube.shape,
                                                    cube.coord_dims('time'))
    return time_weights


def daily_statistics(cube, operator='mean'):
    """
    Compute daily statistics.

    Chunks time in daily periods and computes statistics over them;

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        Daily statistics cube
    """
    if not cube.coords('day_of_year'):
        iris.coord_categorisation.add_day_of_year(cube, 'time')
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    operator = get_iris_analysis_operation(operator)
    cube = cube.aggregated_by(['day_of_year', 'year'], operator)

    cube.remove_coord('day_of_year')
    cube.remove_coord('year')
    return cube


def monthly_statistics(cube, operator='mean'):
    """
    Compute monthly statistics.

    Chunks time in monthly periods and computes statistics over them;

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        Monthly statistics cube
    """
    if not cube.coords('month_number'):
        iris.coord_categorisation.add_month_number(cube, 'time')
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    operator = get_iris_analysis_operation(operator)
    cube = cube.aggregated_by(['month_number', 'year'], operator)
    return cube


def seasonal_statistics(cube, operator='mean'):
    """
    Compute seasonal statistics.

    Chunks time in 3-month periods and computes statistics over them;

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        Seasonal statistic cube
    """
    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    if not cube.coords('season_year'):
        iris.coord_categorisation.add_season_year(cube,
                                                  'time',
                                                  name='season_year')

    operator = get_iris_analysis_operation(operator)

    cube = cube.aggregated_by(['clim_season', 'season_year'], operator)

    # CMOR Units are days so we are safe to operate on days
    # Ranging on [90, 92] days makes this calendar-independent
    def spans_three_months(time):
        """
        Check for three months.

        Parameters
        ----------
        time: iris.DimCoord
            cube time coordinate

        Returns
        -------
        bool
            truth statement if time bounds are 90+2 days.
        """
        return 90 <= (time.bound[1] - time.bound[0]).days <= 92

    three_months_bound = iris.Constraint(time=spans_three_months)
    return cube.extract(three_months_bound)


def annual_statistics(cube, operator='mean'):
    """
    Compute annual statistics.

    Note that this function does not weight the annual mean if
    uneven time periods are present. Ie, all data inside the year
    are treated equally.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        Annual statistics cube
    """
    # TODO: Add weighting in time dimension. See iris issue 3290
    # https://github.com/SciTools/iris/issues/3290

    operator = get_iris_analysis_operation(operator)

    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')
    return cube.aggregated_by('year', operator)


def decadal_statistics(cube, operator='mean'):
    """
    Compute decadal statistics.

    Note that this function does not weight the decadal mean if
    uneven time periods are present. Ie, all data inside the decade
    are treated equally.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        Decadal statistics cube
    """
    # TODO: Add weighting in time dimension. See iris issue 3290
    # https://github.com/SciTools/iris/issues/3290

    operator = get_iris_analysis_operation(operator)

    if not cube.coords('decade'):

        def get_decade(coord, value):
            """Categorize time coordinate into decades."""
            date = coord.units.num2date(value)
            return date.year - date.year % 10

        iris.coord_categorisation.add_categorised_coord(
            cube, 'decade', 'time', get_decade)

    return cube.aggregated_by('decade', operator)


def climate_statistics(cube, operator='mean', period='full'):
    """
    Compute climate statistics with the specified granularity.

    Computes statistics for the whole dataset. It is possible to get them for
    the full period or with the data grouped by day, month or season

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    period: str, optional
        Period to compute the statistic over.
        Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
        'mon', 'daily', 'day'

    Returns
    -------
    iris.cube.Cube
        Monthly statistics cube
    """
    period = period.lower()

    if period in ('full', ):
        operator_method = get_iris_analysis_operation(operator)
        if operator_accept_weights(operator):
            time_weights = get_time_weights(cube)
            if time_weights.min() == time_weights.max():
                # No weighting needed.
                cube = cube.collapsed('time',
                                      operator_method)
            else:
                cube = cube.collapsed('time',
                                      operator_method,
                                      weights=time_weights)
        else:
            cube = cube.collapsed('time', operator_method)
        return cube

    clim_coord = _get_period_coord(cube, period)
    operator = get_iris_analysis_operation(operator)
    clim_cube = cube.aggregated_by(clim_coord, operator)
    clim_cube.remove_coord('time')
    if clim_cube.coord(clim_coord.name()).is_monotonic():
        iris.util.promote_aux_coord_to_dim_coord(clim_cube, clim_coord.name())
    else:
        clim_cube = iris.cube.CubeList(
            clim_cube.slices_over(clim_coord.name())).merge_cube()
    cube.remove_coord(clim_coord)
    return clim_cube


def anomalies(cube, period, reference=None, standardize=False):
    """
    Compute anomalies using a mean with the specified granularity.

    Computes anomalies based on daily, monthly, seasonal or yearly means for
    the full available period

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    period: str
        Period to compute the statistic over.
        Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
        'mon', 'daily', 'day'

    reference: list int, optional, default: None
        Period of time to use a reference, as needed for the 'extract_time'
        preprocessor function
        If None, all available data is used as a reference

    standardize: bool, optional
        If True standardized anomalies are calculated


    Returns
    -------
    iris.cube.Cube
        Anomalies cube
    """
    if reference is None:
        reference_cube = cube
    else:
        reference_cube = extract_time(cube, **reference)
    reference = climate_statistics(reference_cube, period=period)
    if period in ['full']:
        metadata = copy.deepcopy(cube.metadata)
        cube = cube - reference
        cube.metadata = metadata
        if standardize:
            cube_stddev = climate_statistics(
                cube, operator='std_dev', period=period)
            cube = cube / cube_stddev
            cube.units = '1'
        return cube

    cube = _compute_anomalies(cube, reference, period)

    # standardize the results if requested
    if standardize:
        cube_stddev = climate_statistics(cube,
                                         operator='std_dev',
                                         period=period)
        tdim = cube.coord_dims('time')[0]
        reps = cube.shape[tdim] / cube_stddev.shape[tdim]
        if not reps % 1 == 0:
            raise ValueError(
                "Cannot safely apply preprocessor to this dataset, "
                "since the full time period of this dataset is not "
                f"a multiple of the period '{period}'"
            )
        cube.data = cube.core_data() / da.concatenate(
            [cube_stddev.core_data() for _ in range(int(reps))], axis=tdim)
        cube.units = '1'
    return cube


def _compute_anomalies(cube, reference, period):
    cube_coord = _get_period_coord(cube, period)
    ref_coord = _get_period_coord(reference, period)

    data = cube.core_data()
    cube_time = cube.coord('time')
    ref = {}
    for ref_slice in reference.slices_over(ref_coord):
        ref[ref_slice.coord(ref_coord).points[0]] = ref_slice.core_data()

    cube_coord_dim = cube.coord_dims(cube_coord)[0]
    slicer = [slice(None)] * len(data.shape)
    new_data = []
    for i in range(cube_time.shape[0]):
        slicer[cube_coord_dim] = i
        new_data.append(data[tuple(slicer)] - ref[cube_coord.points[i]])
    data = da.stack(new_data, axis=cube_coord_dim)
    cube = cube.copy(data)
    cube.remove_coord(cube_coord)
    return cube


def _get_period_coord(cube, period):
    """Get periods."""
    if period in ['daily', 'day']:
        if not cube.coords('day_of_year'):
            iris.coord_categorisation.add_day_of_year(cube, 'time')
        return cube.coord('day_of_year')
    if period in ['monthly', 'month', 'mon']:
        if not cube.coords('month_number'):
            iris.coord_categorisation.add_month_number(cube, 'time')
        return cube.coord('month_number')
    if period in ['seasonal', 'season']:
        if not cube.coords('season_number'):
            iris.coord_categorisation.add_season_number(cube, 'time')
        return cube.coord('season_number')
    raise ValueError(f"Period '{period}' not supported")


def regrid_time(cube, frequency):
    """
    Align time axis for cubes so they can be subtracted.

    Operations on time units, time points and auxiliary
    coordinates so that any cube from cubes can be subtracted from any
    other cube from cubes. Currently this function supports
    yearly (frequency=yr), monthly (frequency=mon),
    daily (frequency=day), 6-hourly (frequency=6hr),
    3-hourly (frequency=3hr) and hourly (frequency=1hr) data time frequencies.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    frequency: str
        data frequency: mon, day, 1hr, 3hr or 6hr

    Returns
    -------
    iris.cube.Cube
        cube with converted time axis and units.
    """
    # standardize time points
    time_c = [cell.point for cell in cube.coord('time').cells()]
    if frequency == 'yr':
        time_cells = [
            datetime.datetime(t.year, 7, 1, 0, 0, 0) for t in time_c
        ]
    elif frequency == 'mon':
        time_cells = [
            datetime.datetime(t.year, t.month, 15, 0, 0, 0) for t in time_c
        ]
    elif frequency == 'day':
        time_cells = [
            datetime.datetime(t.year, t.month, t.day, 0, 0, 0) for t in time_c
        ]
    elif frequency == '1hr':
        time_cells = [
            datetime.datetime(t.year, t.month, t.day, t.hour, 0, 0)
            for t in time_c
        ]
    elif frequency == '3hr':
        time_cells = [
            datetime.datetime(t.year, t.month, t.day, t.hour - t.hour % 3, 0,
                              0) for t in time_c
        ]
    elif frequency == '6hr':
        time_cells = [
            datetime.datetime(t.year, t.month, t.day, t.hour - t.hour % 6, 0,
                              0) for t in time_c
        ]

    cube.coord('time').points = [
        cube.coord('time').units.date2num(cl)
        for cl in time_cells]

    # uniformize bounds
    cube.coord('time').bounds = None
    cube.coord('time').guess_bounds()

    # remove aux coords that will differ
    reset_aux = ['day_of_month', 'day_of_year']
    for auxcoord in cube.aux_coords:
        if auxcoord.long_name in reset_aux:
            cube.remove_coord(auxcoord)

    # re-add the converted aux coords
    iris.coord_categorisation.add_day_of_month(cube,
                                               cube.coord('time'),
                                               name='day_of_month')
    iris.coord_categorisation.add_day_of_year(cube,
                                              cube.coord('time'),
                                              name='day_of_year')

    return cube


def low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.

    Method borrowed from `iris example
    <https://scitools.org.uk/iris/docs/latest/examples/General/
    SOI_filtering.html?highlight=running%20mean>`_

    Parameters
    ----------
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.

    Returns
    -------
    list:
        List of floats representing the weights.
    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    weights = np.zeros([nwts])
    half_order = nwts // 2
    weights[half_order] = 2 * cutoff
    kidx = np.arange(1., half_order)
    sigma = np.sin(np.pi * kidx / half_order) * half_order / (np.pi * kidx)
    firstfactor = np.sin(2. * np.pi * cutoff * kidx) / (np.pi * kidx)
    weights[(half_order - 1):0:-1] = firstfactor * sigma
    weights[(half_order + 1):-1] = firstfactor * sigma

    return weights[1:-1]


def timeseries_filter(cube, window, span,
                      filter_type='lowpass', filter_stats='sum'):
    """
    Apply a timeseries filter.

    Method borrowed from `iris example
    <https://scitools.org.uk/iris/docs/latest/examples/General/
    SOI_filtering.html?highlight=running%20mean>`_

    Apply each filter using the rolling_window method used with the weights
    keyword argument. A weighted sum is required because the magnitude of
    the weights are just as important as their relative sizes.

    See also the `iris rolling window
    <https://scitools.org.uk/iris/docs/v2.0/iris/iris/
    cube.html#iris.cube.Cube.rolling_window>`_

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    window: int
        The length of the filter window (in units of cube time coordinate).
    span: int
        Number of months/days (depending on data frequency) on which
        weights should be computed e.g. 2-yearly: span = 24 (2 x 12 months).
        Span should have same units as cube time coordinate.
    filter_type: str, optional
        Type of filter to be applied; default 'lowpass'.
        Available types: 'lowpass'.
    filter_stats: str, optional
        Type of statistic to aggregate on the rolling window; default 'sum'.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'

    Returns
    -------
    iris.cube.Cube
        cube time-filtered using 'rolling_window'.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError:
        Cube does not have time coordinate.
    NotImplementedError:
        If filter_type is not implemented.
    """
    try:
        cube.coord('time')
    except iris.exceptions.CoordinateNotFoundError:
        logger.error("Cube %s does not have time coordinate", cube)
        raise

    # Construct weights depending on frequency
    # TODO implement more filters!
    supported_filters = ['lowpass', ]
    if filter_type in supported_filters:
        if filter_type == 'lowpass':
            wgts = low_pass_weights(window, 1. / span)
    else:
        raise NotImplementedError(
            "Filter type {} not implemented, \
            please choose one of {}".format(filter_type,
                                            ", ".join(supported_filters)))

    # Apply filter
    aggregation_operator = get_iris_analysis_operation(filter_stats)
    cube = cube.rolling_window('time',
                               aggregation_operator,
                               len(wgts),
                               weights=wgts)

    return cube
