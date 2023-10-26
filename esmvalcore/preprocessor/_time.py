"""Time operations on cubes.

Allows for selecting data subsets using certain time bounds;
constructing seasonal and area averages.
"""
from __future__ import annotations

import copy
import datetime
import logging
import warnings
from typing import Iterable, Optional
from warnings import filterwarnings

import dask.array as da
import dask.config
import iris
import iris.coord_categorisation
import iris.exceptions
import iris.util
import isodate
import numpy as np
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.time import PartialDateTime

from esmvalcore.cmor.fixes import get_next_month, get_time_bounds
from esmvalcore.iris_helpers import date2num

from ._shared import get_iris_aggregator, update_weights_kwargs

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


def get_time_weights(cube: Cube) -> np.ndarray | da.core.Array:
    """Compute the weighting of the time axis.

    Parameters
    ----------
    cube:
        Input cube.

    Returns
    -------
    np.ndarray or da.core.Array
        Array of time weights for averaging.

    """
    time = cube.coord('time')
    coord_dims = cube.coord_dims('time')

    # Multidimensional time coordinates are not supported: In this case,
    # weights cannot be simply calculated as difference between the bounds
    if len(coord_dims) > 1:
        raise ValueError(
            f"Weighted statistical operations are not supported for "
            f"{len(coord_dims):d}D time coordinates, expected 0D or 1D"
        )

    # Extract 1D time weights (= lengths of time intervals)
    time_weights = time.core_bounds()[:, 1] - time.core_bounds()[:, 0]
    return time_weights


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
    original_dtype = cube.dtype
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

    # Make sure that original dtype is preserved
    new_dtype = clim_cube.dtype
    if original_dtype != new_dtype:
        logger.debug(
            "climate_statistics changed dtype from "
            "%s to %s, changing back", original_dtype, new_dtype)
        clim_cube.data = clim_cube.core_data().astype(original_dtype)

    return clim_cube


def _add_time_weights_coord(cube):
    """Add time weight coordinate to cube (in-place)."""
    time_weights_coord = AuxCoord(
        get_time_weights(cube),
        long_name='_time_weights_',
        units=cube.coord('time').units,
    )
    cube.add_aux_coord(time_weights_coord, cube.coord_dims('time'))


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


def regrid_time(cube: Cube, frequency: str) -> Cube:
    """Align time axis for cubes so they can be subtracted.

    Operations on time units, time points and auxiliary
    coordinates so that any cube from cubes can be subtracted from any
    other cube from cubes. Currently this function supports
    yearly (frequency=yr), monthly (frequency=mon),
    daily (frequency=day), 6-hourly (frequency=6hr),
    3-hourly (frequency=3hr) and hourly (frequency=1hr) data time frequencies.

    Parameters
    ----------
    cube:
        Input cube.
    frequency:
        Data frequency: `mon`, `day`, `1hr`, `3hr` or `6hr`.

    Returns
    -------
    iris.cube.Cube
        Cube with converted time axis and units.

    """
    # standardize time points
    coord = cube.coord('time')
    time_c = coord.units.num2date(coord.points)
    if frequency == 'yr':
        time_cells = [datetime.datetime(t.year, 7, 1, 0, 0, 0) for t in time_c]
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
            datetime.datetime(
                t.year, t.month, t.day, t.hour - t.hour % 3, 0, 0)
            for t in time_c
        ]
    elif frequency == '6hr':
        time_cells = [
            datetime.datetime(
                t.year, t.month, t.day, t.hour - t.hour % 6, 0, 0)
            for t in time_c
        ]

    coord = cube.coord('time')
    cube.coord('time').points = date2num(time_cells, coord.units, coord.dtype)

    # uniformize bounds
    cube.coord('time').bounds = None
    cube.coord('time').bounds = get_time_bounds(cube.coord('time'), frequency)

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
    except iris.exceptions.CoordinateNotFoundError:
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
    if cube_period.total_seconds() / 3600 >= interval:
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
