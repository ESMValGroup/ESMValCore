"""Time operations on cubes.

Allows for selecting data subsets using certain time bounds;
constructing seasonal and area averages.
"""
from __future__ import annotations

import copy
import datetime
import logging
import warnings
from functools import partial
from typing import Iterable, Optional
from warnings import filterwarnings

import dask.array as da
import dask.config
import iris
import iris.coord_categorisation
import iris.util
import isodate
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError
from iris.time import PartialDateTime
from iris.util import broadcast_to_shape
from numpy.typing import DTypeLike

from esmvalcore.cmor.fixes import get_next_month, get_time_bounds
from esmvalcore.iris_helpers import date2num, rechunk_cube
from esmvalcore.preprocessor._shared import (
    get_iris_aggregator,
    get_time_weights,
    preserve_float_dtype,
    update_weights_kwargs,
)

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
        f"Metadata may not be fully descriptive for '{_coord}'.",
        category=UserWarning,
        module='iris',
    )


def extract_time(
    cube: Cube,
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
) -> Cube:
    """Extract a time range from a cube.

    Given a time range passed in as a series of years, months and days, it
    returns a time-extracted cube with data only within the specified
    time range.

    Parameters
    ----------
    cube:
        Input cube.
    start_year:
        Start year.
    start_month:
        Start month.
    start_day:
        Start day.
    end_year:
        End year.
    end_month:
        End month.
    end_day:
        End day.

    Returns
    -------
    iris.cube.Cube
        Sliced cube.

    Raises
    ------
    ValueError
        Time ranges are outside the cube time limits.

    """
    t_1 = PartialDateTime(year=int(start_year),
                          month=int(start_month),
                          day=int(start_day))
    t_2 = PartialDateTime(year=int(end_year),
                          month=int(end_month),
                          day=int(end_day))

    return _extract_datetime(cube, t_1, t_2)


def _parse_start_date(date):
    """Parse start of the input `timerange` tag given in ISO 8601 format.

    Returns a datetime.datetime object.
    """
    if date.startswith('P'):
        start_date = isodate.parse_duration(date)
    else:
        try:
            start_date = isodate.parse_datetime(date)
        except isodate.isoerror.ISO8601Error:
            start_date = isodate.parse_date(date)
            start_date = datetime.datetime.combine(
                start_date, datetime.time.min)
    return start_date


def _parse_end_date(date):
    """Parse end of the input `timerange` given in ISO 8601 format.

    Returns a datetime.datetime object.
    """
    if date.startswith('P'):
        end_date = isodate.parse_duration(date)
    else:
        if len(date) == 4:
            end_date = datetime.datetime(int(date) + 1, 1, 1, 0, 0, 0)
        elif len(date) == 6:
            month, year = get_next_month(int(date[4:]), int(date[0:4]))
            end_date = datetime.datetime(year, month, 1, 0, 0, 0)
        else:
            try:
                end_date = isodate.parse_datetime(date)
            except isodate.ISO8601Error:
                end_date = isodate.parse_date(date)
                end_date = datetime.datetime.combine(end_date,
                                                     datetime.time.min)
            end_date += datetime.timedelta(seconds=1)
    return end_date


def _duration_to_date(duration, reference, sign):
    """Add or subtract a duration period to a reference datetime."""
    date = reference + sign * duration
    return date


def _select_timeslice(cube: Cube, select: np.ndarray) -> Cube | None:
    """Slice a cube along its time axis."""
    if select.any():
        coord = cube.coord('time')
        time_dims = cube.coord_dims(coord)
        if time_dims:
            time_dim = time_dims[0]
            slices = tuple(select if i == time_dim else slice(None)
                           for i in range(cube.ndim))
            cube_slice = cube[slices]
        else:
            cube_slice = cube
    else:
        cube_slice = None
    return cube_slice


def _extract_datetime(
    cube: Cube,
    start_datetime: PartialDateTime,
    end_datetime: PartialDateTime,
) -> Cube:
    """Extract a time range from a cube.

    Given a time range passed in as a datetime.datetime object, it
    returns a time-extracted cube with data only within the specified
    time range with a resolution up to seconds..

    Parameters
    ----------
    cube:
        Input cube.
    start_datetime:
        Start datetime
    end_datetime:
        End datetime

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
        if isinstance(start_datetime.day, int) and start_datetime.day > 30:
            start_datetime.day = 30
        if isinstance(end_datetime.day, int) and end_datetime.day > 30:
            end_datetime.day = 30

    if not cube.coord_dims(time_coord):
        constraint = iris.Constraint(
            time=lambda t: start_datetime <= t.point < end_datetime)
        cube_slice = cube.extract(constraint)
    else:
        # Convert all time points to dates at once, this is much faster
        # than using a constraint.
        dates = time_coord.units.num2date(time_coord.points)
        select = (dates >= start_datetime) & (dates < end_datetime)
        cube_slice = _select_timeslice(cube, select)

    if cube_slice is None:

        def dt2str(time: PartialDateTime) -> str:
            txt = f"{time.year}-{time.month:02d}-{time.day:02d}"
            if any([time.hour, time.minute, time.second]):
                txt += f" {time.hour:02d}:{time.minute:02d}:{time.second:02d}"
            return txt
        raise ValueError(
            f"Time slice {dt2str(start_datetime)} "
            f"to {dt2str(end_datetime)} is outside "
            f"cube time bounds {time_coord.cell(0).point} to "
            f"{time_coord.cell(-1).point}.")

    return cube_slice


def clip_timerange(cube: Cube, timerange: str) -> Cube:
    """Extract time range with a resolution up to seconds.

    Parameters
    ----------
    cube:
        Input cube.
    timerange: str
        Time range in ISO 8601 format.

    Returns
    -------
    iris.cube.Cube
        Sliced cube.

    Raises
    ------
    ValueError
        Time ranges are outside the cube's time limits.

    """
    start_date = _parse_start_date(timerange.split('/')[0])
    end_date = _parse_end_date(timerange.split('/')[1])

    if isinstance(start_date, isodate.duration.Duration):
        start_date = _duration_to_date(start_date, end_date, sign=-1)
    elif isinstance(start_date, datetime.timedelta):
        start_date = _duration_to_date(start_date, end_date, sign=-1)
        start_date -= datetime.timedelta(seconds=1)

    if isinstance(end_date, isodate.duration.Duration):
        end_date = _duration_to_date(end_date, start_date, sign=1)
    elif isinstance(end_date, datetime.timedelta):
        end_date = _duration_to_date(end_date, start_date, sign=1)
        end_date += datetime.timedelta(seconds=1)

    t_1 = PartialDateTime(
        year=start_date.year,
        month=start_date.month,
        day=start_date.day,
        hour=start_date.hour,
        minute=start_date.minute,
        second=start_date.second,
    )

    t_2 = PartialDateTime(
        year=end_date.year,
        month=end_date.month,
        day=end_date.day,
        hour=end_date.hour,
        minute=end_date.minute,
        second=end_date.second,
    )

    return _extract_datetime(cube, t_1, t_2)


def extract_season(cube: Cube, season: str) -> Cube:
    """Slice cube to get only the data belonging to a specific season.

    Parameters
    ----------
    cube:
        Original data
    season:
        Season to extract. Available: DJF, MAM, JJA, SON
        and all sequentially correct combinations: e.g. JJAS

    Returns
    -------
    iris.cube.Cube
        data cube for specified season.

    Raises
    ------
    ValueError
        Requested season is not present in the cube.

    """
    season = season.upper()

    allmonths = 'JFMAMJJASOND' * 2
    if season not in allmonths:
        raise ValueError(f"Unable to extract Season {season} "
                         f"combination of months not possible.")
    sstart = allmonths.index(season)
    res_season = allmonths[sstart + len(season):sstart + 12]
    seasons = [season, res_season]
    coords_to_remove = []

    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube,
                                             'time',
                                             name='clim_season',
                                             seasons=seasons)
        coords_to_remove.append('clim_season')

    if not cube.coords('season_year'):
        iris.coord_categorisation.add_season_year(cube,
                                                  'time',
                                                  name='season_year',
                                                  seasons=seasons)
        coords_to_remove.append('season_year')

    result = cube.extract(iris.Constraint(clim_season=season))
    for coord in coords_to_remove:
        cube.remove_coord(coord)
    if result is None:
        raise ValueError(f'Season {season!r} not present in cube {cube}')
    return result


def extract_month(cube: Cube, month: int) -> Cube:
    """Slice cube to get only the data belonging to a specific month.

    Parameters
    ----------
    cube:
        Original data
    month:
        Month to extract as a number from 1 to 12.

    Returns
    -------
    iris.cube.Cube
        Cube for specified month.

    Raises
    ------
    ValueError
        Requested month is not present in the cube.

    """
    if month not in range(1, 13):
        raise ValueError('Please provide a month number between 1 and 12.')
    if not cube.coords('month_number'):
        iris.coord_categorisation.add_month_number(cube,
                                                   'time',
                                                   name='month_number')
    result = cube.extract(iris.Constraint(month_number=month))
    if result is None:
        raise ValueError(f'Month {month!r} not present in cube {cube}')
    return result


def _aggregate_time_fx(result_cube, source_cube):
    time_dim = set(source_cube.coord_dims(source_cube.coord('time')))
    if source_cube.cell_measures():
        for measure in source_cube.cell_measures():
            measure_dims = set(source_cube.cell_measure_dims(measure))
            if time_dim.intersection(measure_dims):
                logger.debug('Averaging time dimension in measure %s.',
                             measure.var_name)
                result_measure = da.mean(measure.core_data(),
                                         axis=tuple(time_dim))
                measure = measure.copy(result_measure)
                measure_dims = tuple(measure_dims - time_dim)
                result_cube.add_cell_measure(measure, measure_dims)

    if source_cube.ancillary_variables():
        for ancillary_var in source_cube.ancillary_variables():
            ancillary_dims = set(
                source_cube.ancillary_variable_dims(ancillary_var))
            if time_dim.intersection(ancillary_dims):
                logger.debug(
                    'Averaging time dimension in ancillary variable %s.',
                    ancillary_var.var_name)
                result_ancillary_var = da.mean(ancillary_var.core_data(),
                                               axis=tuple(time_dim))
                ancillary_var = ancillary_var.copy(result_ancillary_var)
                ancillary_dims = tuple(ancillary_dims - time_dim)
                result_cube.add_ancillary_variable(ancillary_var,
                                                   ancillary_dims)


@preserve_float_dtype
def hourly_statistics(
    cube: Cube,
    hours: int,
    operator: str = 'mean',
    **operator_kwargs,
) -> Cube:
    """Compute hourly statistics.

    Chunks time in x hours periods and computes statistics over them.

    Parameters
    ----------
    cube:
        Input cube.
    hours:
        Number of hours per period. Must be a divisor of 24, i.e., (1, 2, 3, 4,
        6, 8, 12).
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Hourly statistics cube.

    """
    if not cube.coords('hour_group'):
        iris.coord_categorisation.add_categorised_coord(
            cube,
            'hour_group',
            'time',
            lambda coord, value: coord.units.num2date(value).hour // hours,
            units='1')
    if not cube.coords('day_of_year'):
        iris.coord_categorisation.add_day_of_year(cube, 'time')
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    result = cube.aggregated_by(
        ['hour_group', 'day_of_year', 'year'], agg, **agg_kwargs
    )

    result.remove_coord('hour_group')
    result.remove_coord('day_of_year')
    result.remove_coord('year')

    return result


@preserve_float_dtype
def daily_statistics(
    cube: Cube,
    operator: str = 'mean',
    **operator_kwargs,
) -> Cube:
    """Compute daily statistics.

    Chunks time in daily periods and computes statistics over them;

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Daily statistics cube.

    """
    if not cube.coords('day_of_year'):
        iris.coord_categorisation.add_day_of_year(cube, 'time')
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    result = cube.aggregated_by(['day_of_year', 'year'], agg, **agg_kwargs)

    result.remove_coord('day_of_year')
    result.remove_coord('year')
    return result


@preserve_float_dtype
def monthly_statistics(
    cube: Cube,
    operator: str = 'mean',
    **operator_kwargs,
) -> Cube:
    """Compute monthly statistics.

    Chunks time in monthly periods and computes statistics over them;

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Monthly statistics cube.

    """
    if not cube.coords('month_number'):
        iris.coord_categorisation.add_month_number(cube, 'time')
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    result = cube.aggregated_by(['month_number', 'year'], agg, **agg_kwargs)
    _aggregate_time_fx(result, cube)
    return result


@preserve_float_dtype
def seasonal_statistics(
    cube: Cube,
    operator: str = 'mean',
    seasons: Iterable[str] = ('DJF', 'MAM', 'JJA', 'SON'),
    **operator_kwargs,
) -> Cube:
    """Compute seasonal statistics.

    Chunks time seasons and computes statistics over them.

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    seasons:
        Seasons to build. Available: ('DJF', 'MAM', 'JJA', SON') (default)
        and all sequentially correct combinations holding every month
        of a year: e.g. ('JJAS','ONDJFMAM'), or less in case of prior season
        extraction.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Seasonal statistic cube.

    """
    seasons = tuple(sea.upper() for sea in seasons)

    if any(len(sea) < 2 for sea in seasons):
        raise ValueError(
            f"Minimum of 2 month is required per Seasons: {seasons}.")

    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube,
                                             'time',
                                             name='clim_season',
                                             seasons=seasons)
    else:
        old_seasons = sorted(set(cube.coord('clim_season').points))
        if not all(osea in seasons for osea in old_seasons):
            raise ValueError(
                f"Seasons {seasons} do not match prior season extraction "
                f"{old_seasons}.")

    if not cube.coords('season_year'):
        iris.coord_categorisation.add_season_year(cube,
                                                  'time',
                                                  name='season_year',
                                                  seasons=seasons)

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    result = cube.aggregated_by(
        ['clim_season', 'season_year'], agg, **agg_kwargs
    )

    # CMOR Units are days so we are safe to operate on days
    # Ranging on [29, 31] days makes this calendar-independent
    # the only season this could not work is 'F' but this raises an
    # ValueError
    def spans_full_season(cube: Cube) -> list[bool]:
        """Check for all month present in the season.

        Parameters
        ----------
        cube:
            Input cube.

        Returns
        -------
        list[bool]
            Truth statements if time bounds are within (month*29, month*31)

        """
        time = cube.coord('time')
        num_days = [(tt.bounds[0, 1] - tt.bounds[0, 0]) for tt in time]

        seasons = cube.coord('clim_season').points
        tar_days = [(len(sea) * 29, len(sea) * 31) for sea in seasons]

        return [dt[0] <= dn <= dt[1] for dn, dt in zip(num_days, tar_days)]

    full_seasons = spans_full_season(result)
    result = result[full_seasons]
    _aggregate_time_fx(result, cube)
    return result


@preserve_float_dtype
def annual_statistics(
    cube: Cube,
    operator: str = 'mean',
    **operator_kwargs,
) -> Cube:
    """Compute annual statistics.

    Note that this function does not weight the annual mean if
    uneven time periods are present. Ie, all data inside the year
    are treated equally.

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Annual statistics cube.

    """
    # TODO: Add weighting in time dimension. See iris issue 3290
    # https://github.com/SciTools/iris/issues/3290

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)

    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')
    result = cube.aggregated_by('year', agg, **agg_kwargs)
    _aggregate_time_fx(result, cube)
    return result


@preserve_float_dtype
def decadal_statistics(
    cube: Cube,
    operator: str = 'mean',
    **operator_kwargs,
) -> Cube:
    """Compute decadal statistics.

    Note that this function does not weight the decadal mean if
    uneven time periods are present. Ie, all data inside the decade
    are treated equally.

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Decadal statistics cube.

    """
    # TODO: Add weighting in time dimension. See iris issue 3290
    # https://github.com/SciTools/iris/issues/3290

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)

    if not cube.coords('decade'):

        def get_decade(coord, value):
            """Categorize time coordinate into decades."""
            date = coord.units.num2date(value)
            return date.year - date.year % 10

        iris.coord_categorisation.add_categorised_coord(
            cube, 'decade', 'time', get_decade)
    result = cube.aggregated_by('decade', agg, **agg_kwargs)
    _aggregate_time_fx(result, cube)
    return result


@preserve_float_dtype
def climate_statistics(
    cube: Cube,
    operator: str = 'mean',
    period: str = 'full',
    seasons: Iterable[str] = ('DJF', 'MAM', 'JJA', 'SON'),
    **operator_kwargs,
) -> Cube:
    """Compute climate statistics with the specified granularity.

    Computes statistics for the whole dataset. It is possible to get them for
    the full period or with the data grouped by hour, day, month or season.

    Note
    ----
    The `mean`, `sum` and `rms` operations over the `full` period are weighted
    by the time coordinate, i.e., the length of the time intervals. For `sum`,
    the units of the resulting cube will be multiplied by corresponding time
    units (e.g., days).

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    period:
        Period to compute the statistic over. Available periods: `full`,
        `season`, `seasonal`, `monthly`, `month`, `mon`, `daily`, `day`,
        `hourly`, `hour`, `hr`.
    seasons:
        Seasons to use if needed. Defaults to ('DJF', 'MAM', 'JJA', 'SON').
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Climate statistics cube.
    """
    period = period.lower()

    # Use Cube.collapsed when full period is requested
    if period in ('full', ):
        (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
        agg_kwargs = update_weights_kwargs(
            agg, agg_kwargs, '_time_weights_', cube, _add_time_weights_coord
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=(
                    "Cannot check if coordinate is contiguous: Invalid "
                    "operation for '_time_weights_'"
                ),
                category=UserWarning,
                module='iris',
            )
            clim_cube = cube.collapsed('time', agg, **agg_kwargs)

        # Make sure input and output cubes do not have auxiliary coordinate
        if cube.coords('_time_weights_'):
            cube.remove_coord('_time_weights_')
        if clim_cube.coords('_time_weights_'):
            clim_cube.remove_coord('_time_weights_')

    # Use Cube.aggregated_by for other periods
    else:
        clim_coord = _get_period_coord(cube, period, seasons)
        (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
        clim_cube = cube.aggregated_by(clim_coord, agg, **agg_kwargs)
        clim_cube.remove_coord('time')
        _aggregate_time_fx(clim_cube, cube)
        if clim_cube.coord(clim_coord.name()).is_monotonic():
            iris.util.promote_aux_coord_to_dim_coord(clim_cube,
                                                     clim_coord.name())
        else:
            clim_cube = CubeList(
                clim_cube.slices_over(clim_coord.name())).merge_cube()
        cube.remove_coord(clim_coord)

    return clim_cube


def _add_time_weights_coord(cube):
    """Add time weight coordinate to cube (in-place)."""
    time_weights_coord = AuxCoord(
        get_time_weights(cube),
        long_name='_time_weights_',
        units=cube.coord('time').units,
    )
    cube.add_aux_coord(time_weights_coord, cube.coord_dims('time'))


@preserve_float_dtype
def anomalies(
    cube: Cube,
    period: str,
    reference: Optional[dict] = None,
    standardize: bool = False,
    seasons: Iterable[str] = ('DJF', 'MAM', 'JJA', 'SON'),
) -> Cube:
    """Compute anomalies using a mean with the specified granularity.

    Computes anomalies based on hourly, daily, monthly, seasonal or yearly
    means for the full available period.

    Parameters
    ----------
    cube:
        Input cube.
    period:
        Period to compute the statistic over. Available periods: `full`,
        `season`, `seasonal`, `monthly`, `month`, `mon`, `daily`, `day`,
        `hourly`, `hour`, `hr`.
    reference: optional
        Period of time to use a reference, as needed for the
        :func:`~esmvalcore.preprocessor.extract_time` preprocessor function.
        If ``None``, all available data is used as a reference.
    standardize: optional
        If ``True`` standardized anomalies are calculated.
    seasons: optional
        Seasons to use if needed. Defaults to ('DJF', 'MAM', 'JJA', 'SON').

    Returns
    -------
    iris.cube.Cube
        Anomalies cube.
    """
    if reference is None:
        reference_cube = cube
    else:
        reference_cube = extract_time(cube, **reference)
    reference = climate_statistics(reference_cube,
                                   period=period,
                                   seasons=seasons)
    if period in ['full']:
        metadata = copy.deepcopy(cube.metadata)
        cube = cube - reference
        cube.metadata = metadata
        if standardize:
            cube_stddev = climate_statistics(cube,
                                             operator='std_dev',
                                             period=period,
                                             seasons=seasons)
            cube = cube / cube_stddev
            cube.units = '1'
        return cube

    cube = _compute_anomalies(cube, reference, period, seasons)

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
                f"a multiple of the period '{period}'")
        cube.data = cube.core_data() / da.concatenate(
            [cube_stddev.core_data() for _ in range(int(reps))], axis=tdim)
        cube.units = '1'
    return cube


def _compute_anomalies(
    cube: Cube,
    reference: Cube,
    period: str,
    seasons: Iterable[str],
):
    cube_coord = _get_period_coord(cube, period, seasons)
    ref_coord = _get_period_coord(reference, period, seasons)
    indices = np.empty_like(cube_coord.points, dtype=np.int32)
    for idx, point in enumerate(ref_coord.points):
        indices = np.where(cube_coord.points == point, idx, indices)
    ref_data = reference.core_data()
    axis, = cube.coord_dims(cube_coord)
    if cube.has_lazy_data() and reference.has_lazy_data():
        # Rechunk reference data because iris.cube.Cube.aggregate_by, used to
        # compute the reference, produces very small chunks.
        # https://github.com/SciTools/iris/issues/5455
        ref_chunks = tuple(
            -1 if i == axis else chunk
            for i, chunk in enumerate(cube.lazy_data().chunks)
        )
        ref_data = ref_data.rechunk(ref_chunks)
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        ref_data_broadcast = da.take(ref_data, indices=indices, axis=axis)
    data = cube.core_data() - ref_data_broadcast
    cube = cube.copy(data)
    cube.remove_coord(cube_coord)
    return cube


def _get_period_coord(cube, period, seasons):
    """Get periods."""
    if period in ['hourly', 'hour', 'hr']:
        if not cube.coords('hour'):
            iris.coord_categorisation.add_hour(cube, 'time')
        return cube.coord('hour')
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
            iris.coord_categorisation.add_season_number(cube,
                                                        'time',
                                                        seasons=seasons)
        return cube.coord('season_number')
    raise ValueError(f"Period '{period}' not supported")


def regrid_time(
    cube: Cube,
    frequency: str,
    calendar: Optional[str] = None,
    units: str = 'days since 1850-01-01 00:00:00',
) -> Cube:
    """Align time coordinate for cubes.

    Sets datetimes to common values:

    * Decadal data (e.g., ``frequency='dec'``): 1 January 00:00:00 for the
      given year. Example: 1 January 2005 00:00:00 for given year 2005 (decade
      2000-2010).
    * Yearly data (e.g., ``frequency='yr'``): 1 July 00:00:00 for each year.
      Example: 1 July 1993 00:00:00 for the year 1993.
    * Monthly data (e.g., ``frequency='mon'``): 15th day 00:00:00 for each
      month. Example: 15 October 1993 00:00:00 for the month October 1993.
    * Daily data (e.g., ``frequency='day'``): 12:00:00 for each day. Example:
      14 March 1996 12:00:00 for the day 14 March 1996.
    * `n`-hourly data where `n` is a divisor of 24
      (e.g., ``frequency='3hr'``): center of each time interval. Example:
      03:00:00 for interval 00:00:00-06:00:00 (6-hourly data), 16:30:00 for
      interval 15:00:00-18:00:00 (3-hourly data), or 09:30:00 for interval
      09:00:00-10:00:00 (hourly data).

    The corresponding time bounds will be set according to the rules described
    in :func:`esmvalcore.cmor.fixes.get_time_bounds`. The data type of the new
    time coordinate will be set to `float64` (CMOR default for coordinates).
    Potential auxiliary time coordinates (e.g., `day_of_year`) are also changed
    if present.

    This function does not alter the data in any way.

    Note
    ----
    By default, this will not change the calendar of the input time coordinate.
    For decadal, yearly, and monthly data, it is possible to change the
    calendar using the `calendar` argument. Be aware that changing the calendar
    might introduce (small) errors to your data, especially for extensive
    quantities (those that depend on the period length).

    Parameters
    ----------
    cube:
        Input cube. This input cube will not be modified.
    frequency:
        Data frequency. Allowed are

        * Decadal data (`frequency` must include `dec`, e.g., `dec`)
        * Yearly data (`frequency` must include `yr`, e.g., `yrPt`)
        * Monthly data (`frequency` must include `mon`, e.g., `monC`)
        * Daily data (`frequency` must include `day`, e.g., `day`)
        * `n`-hourly data, where `n` must be a divisor of 24 (`frequency` must
          include `nhr`, e.g., `6hrPt`)
    calendar:
        If given, transform the calendar to the one specified (examples:
        `standard`, `365_day`, etc.). This only works for decadal, yearly and
        monthly data, and will raise an error for other frequencies. If not
        set, the calendar will not be changed.
    units:
        Reference time units used if the calendar of the data is changed.
        Ignored if `calendar` is not set.

    Returns
    -------
    iris.cube.Cube
        Cube with converted time coordinate.

    Raises
    ------
    NotImplementedError
        An invalid `frequency` is given or `calendar` is set for a
        non-supported frequency.

    """
    # Do not overwrite input cube
    cube = cube.copy()
    coord = cube.coord('time')

    # Raise an error if calendar is used for a non-supported frequency
    if calendar is not None and ('day' in frequency or 'hr' in frequency):
        raise NotImplementedError(
            f"Setting a fixed calendar is not supported for frequency "
            f"'{frequency}'"
        )

    # Setup new time coordinate
    new_dates = _get_new_dates(frequency, coord)
    if calendar is not None:
        new_coord = DimCoord(
            coord.points,
            standard_name='time',
            long_name='time',
            var_name='time',
            units=Unit(units, calendar=calendar),
        )
    else:
        new_coord = coord
    new_coord.points = date2num(new_dates, new_coord.units, np.float64)
    new_coord.bounds = get_time_bounds(new_coord, frequency)

    # Replace old time coordinate with new one
    time_dims = cube.coord_dims(coord)
    cube.remove_coord(coord)
    cube.add_dim_coord(new_coord, time_dims)

    # Adapt auxiliary time coordinates if necessary
    aux_coord_names = [
        'day_of_month',
        'day_of_year',
        'hour',
        'month',
        'month_fullname',
        'month_number',
        'season',
        'season_number',
        'season_year',
        'weekday',
        'weekday_fullname',
        'weekday_number',
        'year',
    ]
    for coord_name in aux_coord_names:
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)
            getattr(iris.coord_categorisation, f'add_{coord_name}')(
                cube, new_coord
            )

    return cube


def _get_new_dates(frequency: str, coord: Coord) -> list[datetime.datetime]:
    """Get transformed dates."""
    dates = coord.units.num2date(coord.points)

    if 'dec' in frequency:
        dates = [datetime.datetime(d.year, 1, 1, 0, 0, 0) for d in dates]
    elif 'yr' in frequency:
        dates = [datetime.datetime(d.year, 7, 1, 0, 0, 0) for d in dates]
    elif 'mon' in frequency:
        dates = [
            datetime.datetime(d.year, d.month, 15, 0, 0, 0) for d in dates
        ]
    elif 'day' in frequency:
        dates = [
            datetime.datetime(d.year, d.month, d.day, 12, 0, 0) for d in dates
        ]
    elif 'hr' in frequency:
        (n_hours_str, _, _) = frequency.partition('hr')
        if not n_hours_str:
            n_hours = 1
        else:
            n_hours = int(n_hours_str)
        if 24 % n_hours:
            raise NotImplementedError(
                f"For `n`-hourly data, `n` must be a divisor of 24, got "
                f"'{frequency}'"
            )
        half_interval = datetime.timedelta(hours=n_hours / 2.0)
        dates = [
            datetime.datetime(
                d.year, d.month, d.day, d.hour - d.hour % n_hours, 0, 0
            ) + half_interval
            for d in dates
        ]
    else:
        raise NotImplementedError(f"Frequency '{frequency}' is not supported")

    return dates


def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Method borrowed from `iris example
    <https://scitools-iris.readthedocs.io/en/latest/generated/gallery/general/plot_SOI_filtering.html?highlight=running%20mean>`_

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


@preserve_float_dtype
def timeseries_filter(
    cube: Cube,
    window: int,
    span: int,
    filter_type: str = 'lowpass',
    filter_stats: str = 'sum',
    **operator_kwargs,
) -> Cube:
    """Apply a timeseries filter.

    Method borrowed from `iris example
    <https://scitools-iris.readthedocs.io/en/latest/generated/gallery/general/plot_SOI_filtering.html?highlight=running%20mean>`_

    Apply each filter using the rolling_window method used with the weights
    keyword argument. A weighted sum is required because the magnitude of
    the weights are just as important as their relative sizes.

    See also the iris rolling window :obj:`iris.cube.Cube.rolling_window`.

    Parameters
    ----------
    cube:
        Input cube.
    window:
        The length of the filter window (in units of cube time coordinate).
    span:
        Number of months/days (depending on data frequency) on which
        weights should be computed e.g. 2-yearly: span = 24 (2 x 12 months).
        Span should have same units as cube time coordinate.
    filter_type: optional
        Type of filter to be applied; default 'lowpass'.
        Available types: 'lowpass'.
    filter_stats: optional
        Type of statistic to aggregate on the rolling window; default: `sum`.
        Used to determine the :class:`iris.analysis.Aggregator` object used for
        aggregation. Allowed options are given in :ref:`this table
        <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `filter_stats`.

    Returns
    -------
    iris.cube.Cube
        Cube time-filtered using 'rolling_window'.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError:
        Cube does not have time coordinate.
    NotImplementedError:
        `filter_type` is not implemented.

    """
    try:
        cube.coord('time')
    except CoordinateNotFoundError:
        logger.error("Cube %s does not have time coordinate", cube)
        raise

    # Construct weights depending on frequency
    # TODO implement more filters!
    supported_filters = [
        'lowpass',
    ]
    if filter_type in supported_filters:
        if filter_type == 'lowpass':
            # These weights sum to one and are dimensionless (-> we do NOT need
            # to consider units for sums)
            wgts = low_pass_weights(window, 1. / span)
    else:
        raise NotImplementedError(
            f"Filter type {filter_type} not implemented, "
            f"please choose one of {', '.join(supported_filters)}")

    # Apply filter
    (agg, agg_kwargs) = get_iris_aggregator(filter_stats, **operator_kwargs)
    agg_kwargs['weights'] = wgts
    cube = cube.rolling_window('time', agg, len(wgts), **agg_kwargs)

    return cube


def resample_hours(cube: Cube, interval: int, offset: int = 0) -> Cube:
    """Convert x-hourly data to y-hourly by eliminating extra timesteps.

    Convert x-hourly data to y-hourly (y > x) by eliminating the extra
    timesteps. This is intended to be used only with instantaneous values.

    For example:

    - resample_hours(cube, interval=6): Six-hourly intervals at 0:00, 6:00,
      12:00, 18:00.

    - resample_hours(cube, interval=6, offset=3): Six-hourly intervals at
      3:00, 9:00, 15:00, 21:00.

    - resample_hours(cube, interval=12, offset=6): Twelve-hourly intervals
      at 6:00, 18:00.

    Parameters
    ----------
    cube:
        Input cube.
    interval:
        The period (hours) of the desired data.
    offset: optional
        The firs hour (hours) of the desired data.

    Returns
    -------
    iris.cube.Cube
        Cube with the new frequency.

    Raises
    ------
    ValueError:
        The specified frequency is not a divisor of 24.

    """
    allowed_intervals = (1, 2, 3, 4, 6, 12)
    if interval not in allowed_intervals:
        raise ValueError(
            f'The number of hours must be one of {allowed_intervals}')
    if offset >= interval:
        raise ValueError(f'The offset ({offset}) must be lower than '
                         f'the interval ({interval})')
    time = cube.coord('time')
    cube_period = time.cell(1).point - time.cell(0).point
    if cube_period.total_seconds() / 3600 > interval:
        raise ValueError(f"Data period ({cube_period}) should be lower than "
                         f"the interval ({interval})")
    hours = [PartialDateTime(hour=h) for h in range(0 + offset, 24, interval)]
    dates = time.units.num2date(time.points)
    select = np.zeros(len(dates), dtype=bool)
    for hour in hours:
        select |= dates == hour
    cube = _select_timeslice(cube, select)
    if cube is None:
        raise ValueError(
            f"Time coordinate {dates} does not contain {hours} for {cube}")

    return cube


def resample_time(
    cube: Cube,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour: Optional[int] = None,
) -> Cube:
    """Change frequency of data by resampling it.

    Converts data from one frequency to another by extracting the timesteps
    that match the provided month, day and/or hour. This is meant to be used
    with instantaneous values when computing statistics is not desired.

    For example:

    - resample_time(cube, hour=6): Daily values taken at 6:00.

    - resample_time(cube, day=15, hour=6): Monthly values taken at 15th
      6:00.

    - resample_time(cube, month=6): Yearly values, taking in June

    - resample_time(cube, month=6, day=1): Yearly values, taking 1st June

    The condition must yield only one value per interval: the last two samples
    above will produce yearly data, but the first one is meant to be used to
    sample from monthly output and the second one will work better with daily.

    Parameters
    ----------
    cube:
        Input cube.
    month: optional
        Month to extract.
    day: optional
        Day to extract.
    hour: optional
        Hour to extract.

    Returns
    -------
    iris.cube.Cube
        Cube with the new frequency.

    """
    time = cube.coord('time')
    dates = time.units.num2date(time.points)
    requested = PartialDateTime(month=month, day=day, hour=hour)
    select = dates == requested
    cube = _select_timeslice(cube, select)
    if cube is None:
        raise ValueError(
            f"Time coordinate {dates} does not contain {requested} for {cube}")
    return cube


def _lin_pad(array: np.ndarray, delta: float, pad_with: int) -> np.ndarray:
    """Linearly pad an array on both sides with constant difference."""
    end_values = (array[0] - pad_with * delta, array[-1] + pad_with * delta)
    new_array = np.pad(array, pad_with, 'linear_ramp', end_values=end_values)
    return new_array


def _guess_time_bounds(time_coord: DimCoord) -> None:
    """Guess bounds of time coordinate in-place."""
    if time_coord.has_bounds():
        return
    try:
        time_coord.guess_bounds()
    except ValueError:  # coordinate has only 1 point
        point = time_coord.points[0]
        time_coord.bounds = [[point - 0.5, point + 0.5]]


def _get_lst_offset(lon_coord: Coord) -> np.ndarray:
    """Get offsets to shift UTC time to local solar time (LST).

    Note
    ----
    This function expects longitude in degrees. Can be in [0, 360] or [-180,
    180] format.

    """
    # Make sure that longitude is in degrees and shift it to [-180, 180] first
    # (do NOT overwrite input coordinate)
    lon_coord = lon_coord.copy()
    lon_coord.convert_units('degrees')
    shifted_lon = (lon_coord.points + 180.0) % 360 - 180.0
    return 12.0 * (shifted_lon / 180.0)


def _get_lsts(time_coord: DimCoord, lon_coord: Coord) -> np.ndarray:
    """Get array of binned local solar times (LSTs) of shape (lon, time).

    Note
    ----
    LSTs outside of the time bins given be the time coordinate bounds are put
    into a bin below/above the time coordinate.

    """
    # Pad time coordinate with 1 time step at both sides for the bins for LSTs
    # outside of the time coordinate
    dtime = np.abs(
        time_coord.bounds[0, 1] - time_coord.bounds[0, 0]
    )
    new_points = _lin_pad(time_coord.points, dtime, 1)
    bnds = time_coord.bounds
    new_bounds = np.stack(
        (_lin_pad(bnds[:, 0], dtime, 1), _lin_pad(bnds[:, 1], dtime, 1)),
        axis=-1,
    )
    time_coord = time_coord.copy(new_points, bounds=new_bounds)

    n_time = time_coord.shape[0]
    n_lon = lon_coord.shape[0]

    # Calculate LST
    time_array = np.broadcast_to(time_coord.points, (n_lon, n_time))
    time_offsets = _get_lst_offset(lon_coord).reshape(-1, 1)
    exact_lst_array = time_array + time_offsets  # (lon, time)

    # Put LST into bins given be the time coordinate bounds
    bins = np.concatenate(([time_coord.bounds[0, 0]], time_coord.bounds[:, 1]))
    idx = np.digitize(exact_lst_array, bins) - 1  # (lon, time); idx for time
    idx[idx < 0] = 0  # values outside the time coordinate
    idx[idx >= n_time] = - 1  # values outside the time coordinate
    lst_array = time_coord.points[idx]  # (lon, time)

    # Remove time steps again that have been added previously
    lst_array = lst_array[:, 1:-1]

    return lst_array


def _get_time_index_and_mask(
    time_coord: DimCoord,
    lon_coord: Coord,
) -> tuple[np.ndarray, np.ndarray]:
    """Get advanced index and mask for time dimension of shape (time, lon).

    Note
    ----
    The mask considers the fact that not all values for all local solar times
    (LSTs) are given.  E.g., for hourly data with first time point 01:00:00
    UTC, LST in Berlin is already 02:00:00 (assuming no daylight saving time).
    Thus, for 01:00:00 LST on this day, there is no value for Berlin.

    """
    # Make sure that time coordinate has bounds (these are necessary for the
    # binning) and uses 'hours' as reference units
    time_coord.convert_units(
        Unit('hours since 1850-01-01', calendar=time_coord.units.calendar)
    )
    _guess_time_bounds(time_coord)

    lsts = _get_lsts(time_coord, lon_coord)  # (lon, time)
    n_time = time_coord.points.shape[0]

    # We use np.searchsorted to calculate the indices necessary to put the UTC
    # times into their corresponding (binned) LSTs. These incides are 2D since
    # they depend on time and longitude.
    searchsorted_l = partial(np.searchsorted, side='left')
    _get_indices_l = np.vectorize(searchsorted_l, signature='(i),(i)->(i)')
    time_index_l = _get_indices_l(lsts, time_coord.points)  # (lon, time)

    # To calculate the mask, we need to detect which LSTs are outside of the
    # time coordinate. Unfortunately, searchsorted can only detect outliers on
    # one side of the array. This side is determined by the `side` keyword
    # argument. To consistently detect outliers on both sides, we use
    # searchsorted again, this time with `side='right'` (the default is
    # 'left'). Indices that are the same in both arrays need to be masked, as
    # these are the ones outside of the time coordinate. All others will
    # change.
    searchsorted_r = partial(np.searchsorted, side='right')
    _get_indices_r = np.vectorize(searchsorted_r, signature='(i),(i)->(i)')
    time_index_r = _get_indices_r(lsts, time_coord.points)  # (lon, time)
    mask = time_index_l == time_index_r  # (lon, time)

    # The index is given by the left indices (these are identical to the right
    # indices minus 1)
    time_index_l[time_index_l < 0] = 0  # will be masked
    time_index_l[time_index_l >= n_time] = -1  # will be masked

    return (time_index_l.T, mask.T)  # (time, lon)


def _transform_to_lst_eager(
    data: np.ndarray,
    time_index: np.ndarray,
    mask: np.ndarray,
    *,
    time_dim: int,
    lon_dim: int,
    **__,
) -> np.ndarray:
    """Transform array with UTC coord to local solar time (LST) coord.

    Note
    ----
    This function is the eager version of `_transform_to_lst_lazy`.

    `data` needs to be at least 2D. `time_dim` and `lon_dim` correspond to the
    dimensions that describe time and longitude dimensions in `data`,
    respectively.

    `time_index` is an `advanced index
    <https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing>`__
    for the time dimension of `data` with shape (time, lon). It is used to
    reorder the data along the time axis based on the longitude axis.

    `mask` is 2D with shape (time, lon) that will be applied to the final data.

    """
    # Apart from the time index, all other dimensions will stay the same; this
    # is ensured with np.ogrid
    idx = np.ogrid[tuple(slice(0, d) for d in data.shape)]
    time_index = broadcast_to_shape(
        time_index, data.shape, (time_dim, lon_dim)
    )
    idx[time_dim] = time_index
    new_data = data[tuple(idx)]

    # Apply properly broadcasted mask
    mask = broadcast_to_shape(mask, new_data.shape, (time_dim, lon_dim))
    new_mask = mask | np.ma.getmaskarray(new_data)

    return np.ma.masked_array(new_data, mask=new_mask)


def _transform_to_lst_lazy(
    data: da.core.Array,
    time_index: np.ndarray,
    mask: np.ndarray,
    *,
    time_dim: int,
    lon_dim: int,
    output_dtypes: DTypeLike,
) -> da.core.Array:
    """Transform array with UTC coord to local solar time (LST) coord.

    Note
    ----
    This function is the lazy version of `_transform_to_lst_eager` using
    dask's :func:`dask.array.apply_gufunc`.

    `data` needs to be at least 2D. `time_dim` and `lon_dim` correspond to the
    dimensions that describe time and longitude dimensions in `data`,
    respectively.

    `time_index` is an `advanced index
    <https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing>`__
    for the time dimension of `data` with shape (time, lon). It is used to
    reorder the data along the time axis based on the longitude axis.

    `mask` is 2D with shape (time, lon) that will be applied to the final data.

    """
    new_data = da.apply_gufunc(
        _transform_to_lst_eager,
        '(t,x),(t,x),(t,x)->(t,x)',
        data,
        time_index,
        mask,
        axes=[(time_dim, lon_dim), (0, 1), (0, 1), (time_dim, lon_dim)],
        output_dtypes=output_dtypes,
        time_dim=-2,  # this is ensured by da.apply_gufunc
        lon_dim=-1,  # this is ensured by da.apply_gufunc
    )
    return new_data


def _transform_arr_to_lst(
    data: np.ndarray | da.core.Array,
    time_index: np.ndarray,
    mask: np.ndarray,
    *,
    time_dim: int,
    lon_dim: int,
    output_dtypes: DTypeLike,
) -> np.ndarray | da.core.Array:
    """Transform array with UTC coord to local solar time (LST) coord.

    Note
    ----
    This function either calls `_transform_to_lst_eager` or
    `_transform_to_lst_lazy` depending on the type of input data.

    """
    if isinstance(data, np.ndarray):
        func = _transform_to_lst_eager  # type: ignore
    else:
        func = _transform_to_lst_lazy  # type: ignore
    new_data = func(
        data,  # type: ignore
        time_index,
        mask,
        time_dim=time_dim,
        lon_dim=lon_dim,
        output_dtypes=output_dtypes,
    )
    return new_data


def _transform_cube_to_lst(cube: Cube) -> Cube:
    """Transform cube to local solar time (LST) coordinate (lazy; in-place)."""
    # Rechunk cube properly (it must not be chunked along time and longitude
    # dimension); this also creates a new cube so the original input cube is
    # not overwritten
    complete_coords = [
        cube.coord('time', dim_coords=True), cube.coord('longitude'),
    ]
    cube = rechunk_cube(cube, complete_coords)

    time_coord = cube.coord('time', dim_coords=True)
    lon_coord = cube.coord('longitude')
    time_dim = cube.coord_dims(time_coord)[0]
    lon_dim = cube.coord_dims(lon_coord)[0]

    # Transform cube data
    (time_index, mask) = _get_time_index_and_mask(time_coord, lon_coord)
    cube.data = _transform_arr_to_lst(
        cube.core_data(),
        time_index,
        mask,
        time_dim=time_dim,
        lon_dim=lon_dim,
        output_dtypes=cube.dtype,
    )

    # Transform aux coords that span time and longitude dimensions
    for coord in cube.coords(dim_coords=False):
        dims = cube.coord_dims(coord)
        if time_dim in dims and lon_dim in dims:
            time_dim_ = dims.index(time_dim)
            lon_dim_ = dims.index(lon_dim)
            coord.points = _transform_arr_to_lst(
                coord.core_points(),
                time_index,
                mask,
                time_dim=time_dim_,
                lon_dim=lon_dim_,
                output_dtypes=coord.dtype,
            )
            if coord.has_bounds():
                coord.bounds = _transform_arr_to_lst(
                    coord.core_bounds(),
                    time_index,
                    mask,
                    time_dim=time_dim_,
                    lon_dim=lon_dim_,
                    output_dtypes=coord.bounds_dtype,
                )

    # Transform cell measures that span time and longitude dimensions
    for cell_measure in cube.cell_measures():
        dims = cube.cell_measure_dims(cell_measure)
        if time_dim in dims and lon_dim in dims:
            time_dim_ = dims.index(time_dim)
            lon_dim_ = dims.index(lon_dim)
            cell_measure.data = _transform_arr_to_lst(
                cell_measure.core_data(),
                time_index,
                mask,
                time_dim=time_dim_,
                lon_dim=lon_dim_,
                output_dtypes=cell_measure.dtype,
            )

    # Transform ancillary variables that span time and longitude dimensions
    for anc_var in cube.ancillary_variables():
        dims = cube.ancillary_variable_dims(anc_var)
        if time_dim in dims and lon_dim in dims:
            time_dim_ = dims.index(time_dim)
            lon_dim_ = dims.index(lon_dim)
            anc_var.data = _transform_arr_to_lst(
                anc_var.core_data(),
                time_index,
                mask,
                time_dim=time_dim_,
                lon_dim=lon_dim_,
                output_dtypes=anc_var.dtype,
            )

    return cube


def _check_cube_coords(cube):
    if not cube.coords('time', dim_coords=True):
        raise CoordinateNotFoundError(
            f"Input cube {cube.summary(shorten=True)} needs a dimensional "
            f"coordinate `time`"
        )
    time_coord = cube.coord('time', dim_coords=True)
    # The following works since DimCoords are always 1D and monotonic
    if time_coord.points[0] > time_coord.points[-1]:
        raise ValueError("`time` coordinate must be monotonically increasing")

    if not cube.coords('longitude'):
        raise CoordinateNotFoundError(
            f"Input cube {cube.summary(shorten=True)} needs a coordinate "
            f"`longitude`"
        )
    lon_ndim = len(cube.coord_dims('longitude'))
    if lon_ndim != 1:
        raise CoordinateMultiDimError(
            f"Input cube {cube.summary(shorten=True)} needs a 1D coordinate "
            f"`longitude`, got {lon_ndim:d}D"
        )


@preserve_float_dtype
def local_solar_time(cube: Cube) -> Cube:
    """Convert UTC time coordinate to local solar time (LST).

    This preprocessor transforms input data with a UTC-based time coordinate to
    a `local solar time (LST) <https://en.wikipedia.org/wiki/Solar_time>`__
    coordinate. In LST, 12:00 noon is defined as the moment when the sun
    reaches its highest point in the sky. Thus, LST is mainly determined by
    longitude of a location. LST is particularly suited to analyze diurnal
    cycles across larger regions of the globe, which would be phase-shifted
    against each other when using UTC time.

    To transform data from UTC to LST, this function shifts data along the time
    axis based on the longitude. In addition to the `cube`'s data, this
    function also considers auxiliary coordinates, cell measures and ancillary
    variables that span both the time and longitude dimension.

    Note
    ----
    This preprocessor preserves the temporal frequency of the input data. For
    example, hourly input data will be transformed into hourly output data. For
    this, a location's exact LST will be put into corresponding bins defined
    by the bounds of the input time coordinate (in this example, the bin size
    is 1 hour). If time bounds are not given or cannot be approximated (only
    one time step is given), a bin size of 1 hour is assumed.

    LST is approximated as `UTC_time + 12*longitude/180`, where `longitude` is
    assumed to be in [-180, 180] (this function automatically calculates the
    correct format for the longitude). This is only an approximation since the
    exact LST also depends on the day of year due to the eccentricity of
    Earth's orbit (see `equation of time
    <https://en.wikipedia.org/wiki/Equation_of_time>`__). However, since the
    corresponding error is ~15 min at most, this is ignored here, as most
    climate model data has a courser temporal resolution and the time scale for
    diurnal evolution of meteorological phenomena is usually in the order of
    hours, not minutes.

    Parameters
    ----------
    cube:
        Input cube. Needs a 1D monotonically increasing dimensional coordinate
        `time` (assumed to refer to UTC time) and a 1D coordinate `longitude`.

    Returns
    -------
    Cube
        Transformed cube of same shape as input cube with an LST coordinate
        instead of UTC time.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        Input cube does not contain valid `time` and/or `longitude` coordinate.
    iris.exceptions.CoordinateMultiDimError
        Input cube has multidimensional `longitude` coordinate.
    ValueError
        `time` coordinate of input cube is not monotonically increasing.

    """
    # Make sure that cube has valid time and longitude coordinates
    _check_cube_coords(cube)

    # Transform cube data and all dimensional metadata that spans time AND
    # longitude dimensions
    cube = _transform_cube_to_lst(cube)

    # Adapt metadata of time coordinate
    cube.coord('time', dim_coords=True).long_name = 'Local Solar Time'

    return cube
