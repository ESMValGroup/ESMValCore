"""Time operations on cubes.

Allows for selecting data subsets using certain time bounds;
constructing seasonal and area averages.
"""
import datetime
import logging

import cf_units
import iris
import iris.coord_categorisation
import numpy as np

from ._shared import get_iris_analysis_operation, operator_accept_weights

logger = logging.getLogger(__name__)


def extract_time(cube,
                 start_year,
                 start_month,
                 start_day,
                 end_year,
                 end_month,
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
    start_date = datetime.datetime(
        int(start_year), int(start_month), int(start_day))
    end_date = datetime.datetime(int(end_year), int(end_month), int(end_day))

    t_1 = time_units.date2num(start_date)
    t_2 = time_units.date2num(end_date)
    constraint = iris.Constraint(
        time=lambda t: t_1 < time_units.date2num(t.point) < t_2)

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
        iris.coord_categorisation.add_season_year(
            cube, 'time', name='season_year')
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
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

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
    return cube
    

def monthly_statistics(cube, operator='mean'):
    """
    Compute monthly statistics.

    Chunks time in monthly periods and computes means over them;

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

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

    Chunks time in 3-month periods and computes means over them;

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

    Returns
    -------
    iris.cube.Cube
        Seasonal statistic cube
    """
    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    if not cube.coords('season_year'):
        iris.coord_categorisation.add_season_year(
            cube, 'time', name='season_year')

    operator = get_iris_analysis_operation(operator)

    cube = cube.aggregated_by(['clim_season', 'season_year'],
                              operator)

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
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

    Returns
    -------
    iris.cube.Cube
        Annual mean cube
    """
    # time_weights = get_time_weights(cube)

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
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

    Returns
    -------
    iris.cube.Cube
        Annual mean cube
    """
    # time_weights = get_time_weights(cube)

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

    Computes daily, monthly, seasonal or yearly statistics for the
    full available period

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'variance', 'min',
        'max'

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
            cube = cube.collapsed(
                'time', operator_method, weights=time_weights
            )
        else:
            cube = cube.collapsed('time', operator_method)
        return cube

    clim_coord = 'clim_coord'
    if period in ['daily', 'day']:
        iris.coord_categorisation.add_day_of_year(
            cube, 'time', name=clim_coord
        )
    elif period in ['monthly', 'month', 'mon']:
        iris.coord_categorisation.add_month_number(
            cube, 'time', name=clim_coord
        )
    elif period in ['seasonal', 'season']:
        iris.coord_categorisation.add_season(
            cube, 'time', name=clim_coord
        )
    else:
        raise ValueError(
            'Climate_statistics does not support period %s' % period
        )
    operator = get_iris_analysis_operation(operator)
    cube = cube.aggregated_by(clim_coord, operator)
    cube.remove_coord(clim_coord)
    return cube


def regrid_time(cube, frequency):
    """
    Align time axis for cubes so they can be subtracted.

    Operations on time units, calendars, time points and auxiliary
    coordinates so that any cube from cubes can be subtracted from any
    other cube from cubes. Currently this function supports only monthly
    (frequency=mon) and daily (frequency=day) data time frequencies.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    frequency: str
        data frequency: mon or day

    Returns
    -------
    iris.cube.Cube
        cube with converted time axis and units.
    """
    # fix calendars
    cube.coord('time').units = cf_units.Unit(
        cube.coord('time').units.origin,
        calendar='gregorian',
    )

    # standardize time points
    time_c = [cell.point for cell in cube.coord('time').cells()]
    if frequency == 'mon':
        cube.coord('time').cells = [
            datetime.datetime(t.year, t.month, 15, 0, 0, 0) for t in time_c
        ]
    elif frequency == 'day':
        cube.coord('time').cells = [
            datetime.datetime(t.year, t.month, t.day, 0, 0, 0) for t in time_c
        ]
    # TODO add correct handling of hourly data
    # this is a bit more complicated since it can be 3h, 6h etc
    cube.coord('time').points = [
        cube.coord('time').units.date2num(cl)
        for cl in cube.coord('time').cells
    ]

    # uniformize bounds
    cube.coord('time').bounds = None
    cube.coord('time').guess_bounds()

    # remove aux coords that will differ
    reset_aux = ['day_of_month', 'day_of_year']
    for auxcoord in cube.aux_coords:
        if auxcoord.long_name in reset_aux:
            cube.remove_coord(auxcoord)

    # re-add the converted aux coords
    iris.coord_categorisation.add_day_of_month(
        cube, cube.coord('time'), name='day_of_month')
    iris.coord_categorisation.add_day_of_year(
        cube, cube.coord('time'), name='day_of_year')

    return cube