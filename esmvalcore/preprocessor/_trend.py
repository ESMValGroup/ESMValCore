"""Preprocessor functions calculate trends from data."""
import logging

import dask.array as da
import iris
import numpy as np
from cf_units import Unit

logger = logging.getLogger(__name__)


def _mask_xy(x_arr, y_arr):
    """Mask X and Y arrays."""
    if np.ma.is_masked(y_arr):
        x_arr = x_arr[~y_arr.mask]
        y_arr = y_arr[~y_arr.mask]
    return (x_arr, y_arr)


def _slope(x_arr, y_arr):
    """Calculate slope."""
    xy_sum = (x_arr * y_arr).sum()
    xx_sum = (x_arr * x_arr).sum()
    x_mean = x_arr.mean()
    slope = (xy_sum - x_mean * y_arr.sum()) / (xx_sum - x_mean * x_arr.sum())
    return slope


def _get_slope(y_arr, x_arr):
    """Calculate slope between X and Y array."""
    (x_arr, y_arr) = _mask_xy(x_arr, y_arr)

    # If less than 2 points, slope calculation not possible
    if len(y_arr) < 2:
        return np.nan

    # Calculate slope
    return _slope(x_arr, y_arr)


def _get_slope_stderr(y_arr, x_arr):
    """Calculate standard error of slope between X and Y array."""
    (x_arr, y_arr) = _mask_xy(x_arr, y_arr)

    # If less than 2 points, slope stderr calculation not possible
    if len(y_arr) < 2:
        return np.nan

    # Calculate standard error of slope
    if len(y_arr) == 2:
        return 0.0
    dof = len(y_arr) - 2
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    slope = _slope(x_arr, y_arr)
    intercept = y_mean - slope * x_mean
    y_estim = slope * x_arr + intercept
    slope_stderr = np.sqrt(((y_arr - y_estim)**2).sum() / dof /
                           ((x_arr - x_mean)**2).sum())
    return slope_stderr


def _set_trend_units(cube, coord):
    """Set correct trend units for cube."""
    coord_units = coord.units
    if coord_units.is_time_reference():
        coord_units = Unit(coord_units.symbol.split()[0])
    invalid_units = any([cube.units is None, cube.units.is_unknown(),
                         cube.units.is_no_unit(), coord_units.is_no_unit()])
    if not invalid_units:
        cube.units /= coord_units


def linear_trend(cube, coordinate='time'):
    """Calculate linear trend of data along a given coordinate.

    The linear trend is defined as the slope of an ordinary linear regression.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input data.
    coordinate : str, optional (default: 'time')
        Dimensional coordinate over which the trend is calculated.

    Returns
    -------
    iris.cube.Cube
        Trends.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        The dimensional coordinate with the name ``coordinate`` is not found in
        ``cube``.

    """
    coord = cube.coord(coordinate, dim_coords=True)

    # Construct aggregator and calculate trend
    def call_func(data, axis, x_data):
        """Calculate trend."""
        trend_arr = np.apply_along_axis(_get_slope, axis, data, x_data)
        trend_arr = np.ma.masked_invalid(trend_arr)
        return trend_arr

    def lazy_func(data, axis, x_data):
        """Calculate trend lazily."""
        trend_arr = da.apply_along_axis(
            _get_slope, axis, data, x_data, dtype=data.dtype, shape=())
        trend_arr = da.ma.masked_invalid(trend_arr)
        return trend_arr

    aggregator = iris.analysis.Aggregator('trend', call_func,
                                          lazy_func=lazy_func,
                                          x_data=coord.points)
    cube = cube.collapsed(coord, aggregator)

    # Adapt units
    _set_trend_units(cube, coord)

    return cube


def linear_trend_stderr(cube, coordinate='time'):
    """Calculate standard error of linear trend along a given coordinate.

    This gives the standard error (not confidence intervals!) of the trend
    defined as the standard error of the estimated slope of an ordinary linear
    regression.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input data.
    coordinate : str, optional (default: 'time')
        Dimensional coordinate over which the standard error of the trend is
        calculated.

    Returns
    -------
    iris.cube.Cube
        Standard errors of trends.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        The dimensional coordinate with the name ``coordinate`` is not found in
        ``cube``.

    """
    coord = cube.coord(coordinate, dim_coords=True)

    # Construct aggregator and calculate standard error of the trend
    def call_func(data, axis, x_data):
        """Calculate trend standard error."""
        trend_std_arr = np.apply_along_axis(_get_slope_stderr, axis, data,
                                            x_data)
        trend_std_arr = np.ma.masked_invalid(trend_std_arr)
        return trend_std_arr

    def lazy_func(data, axis, x_data):
        """Calculate trend standard error lazily."""
        trend_std_arr = da.apply_along_axis(
            _get_slope_stderr, axis, data, x_data, dtype=data.dtype, shape=())
        trend_std_arr = da.ma.masked_invalid(trend_std_arr)
        return trend_std_arr

    aggregator = iris.analysis.Aggregator('trend_stderr', call_func,
                                          lazy_func=lazy_func,
                                          x_data=coord.points)
    cube = cube.collapsed(coord, aggregator)

    # Adapt units
    _set_trend_units(cube, coord)

    return cube
