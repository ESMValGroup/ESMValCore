"""Time operations on cubes.

Allows for selecting data subsets using certain time bounds;
constructing seasonal and area averages.
"""
import datetime
import logging
from warnings import filterwarnings

import dask.array as da
import iris
import iris.coord_categorisation
import iris.util
import numpy as np

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

    Given a time range passed in as a series of years, mnoths and days, it
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
    time_units = cube.coord('time').units
    if time_units.calendar == '360_day':
        if start_day > 30:
            start_day = 30
        if end_day > 30:
            end_day = 30
    start_date = datetime.datetime(int(start_year), int(start_month),
                                   int(start_day))
    end_date = datetime.datetime(int(end_year), int(end_month), int(end_day))

    t_1 = time_units.date2num(start_date)
    t_2 = time_units.date2num(end_date)
    constraint = iris.Constraint(
        time=lambda t: t_1 <= time_units.date2num(t.point) < t_2)

    cube_slice = cube.extract(constraint)
    if cube_slice is None:
        start_cube = str(cube.coord('time').points[0])
        end_cube = str(cube.coord('time').points[-1])
        raise ValueError(
            f"Time slice {start_date} to {end_date} is outside cube "
            f"time bounds {start_cube} to {end_cube}.")

    # Issue when time dimension was removed when only one point as selected.
    if cube_slice.ndim != cube.ndim:
        time_1 = cube.coord('time')
        time_2 = cube_slice.coord('time')
        if time_1 == time_2:
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
    time_thickness = time.bounds[..., 1] - time.bounds[..., 0]

    # The weights need to match the dimensionality of the cube.
    slices = [None for i in cube.shape]
    coord_dim = cube.coord_dims('time')[0]
    slices[coord_dim] = slice(None)
    time_thickness = np.abs(time_thickness[tuple(slices)])
    ones = np.ones_like(cube.data)
    time_weights = time_thickness * ones
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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
            """Callback function to get decades from cube."""
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
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'

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
            cube = cube.collapsed('time',
                                  operator_method,
                                  weights=time_weights)
        else:
            cube = cube.collapsed('time', operator_method)
        return cube

    clim_coord = _get_period_coord(cube, period)
    operator = get_iris_analysis_operation(operator)
    clim_cube = cube.aggregated_by(clim_coord, operator)
    cube.remove_coord(clim_coord)
    clim_cube.remove_coord('time')
    iris.util.promote_aux_coord_to_dim_coord(clim_cube, clim_coord.name())
    return clim_cube


def anomalies(cube, period):
    """
    Compute anomalies using a mean with the specified granularity.

    Computes anomalies based on daily, monthly, seasonal or yearly means for
    the full available period

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    period: str, optional
        Period to compute the statistic over.
        Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
        'mon', 'daily', 'day'

    Returns
    -------
    iris.cube.Cube
        Monthly statistics cube
    """
    reference = climate_statistics(cube, period=period)
    if period in ['full']:
        return cube - reference

    cube_coord = _get_period_coord(cube, period)
    ref_coord = _get_period_coord(reference, period)

    data = cube.core_data()
    cube_time = cube.coord('time')
    ref = {}
    for ref_slice in reference.slices_over(ref_coord):
        ref[ref_slice.coord(ref_coord).points[0]] = da.ravel(
            ref_slice.core_data())
    cube_coord_dim = cube.coord_dims(cube_coord)[0]
    for i in range(cube_time.shape[0]):
        time = cube_time.points[i]
        indexes = cube_time.points == time
        indexes = iris.util.broadcast_to_shape(indexes, data.shape,
                                               (cube_coord_dim, ))
        data[indexes] = data[indexes] - ref[cube_coord.points[i]]

    cube = cube.copy(data)
    return cube


def _get_period_coord(cube, period):
    if period in ['daily', 'day']:
        if not cube.coords('day_of_year'):
            iris.coord_categorisation.add_day_of_year(cube, 'time')
        return cube.coord('day_of_year')
    elif period in ['monthly', 'month', 'mon']:
        if not cube.coords('month_number'):
            iris.coord_categorisation.add_month_number(cube, 'time')
        return cube.coord('month_number')
    elif period in ['seasonal', 'season']:
        if not cube.coords('season_number'):
            iris.coord_categorisation.add_season_number(cube, 'time')
        return cube.coord('season_number')
    raise ValueError('Period %s not supported')


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
