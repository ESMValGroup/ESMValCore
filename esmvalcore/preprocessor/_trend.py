"""Preprocessor functions calculate trends from data."""
import logging

import dask.array as da
import iris
import numpy as np
from cf_units import Unit

logger = logging.getLogger(__name__)


def _remove_axis_np(data, axis=None):
    """Non-lazy aggregator function to remove an axis."""
    return np.take(data, 0, axis=axis)


def _remove_axis_da(data, axis=None):
    """Lazy aggregator function to remove an axis."""
    return da.take(data, 0, axis=axis)


def _mask_xy(x_arr, y_arr):
    """Mask X and Y arrays."""
    if np.ma.is_masked(y_arr):
        x_arr = x_arr[~y_arr.mask]
        y_arr = y_arr[~y_arr.mask]
    return (x_arr, y_arr)


def _slope(x_arr, y_arr):
    """Calculate slope."""
    x_demean = x_arr - x_arr.mean()
    y_demean = y_arr - y_arr.mean()
    slope = (x_demean * y_demean).sum() / (x_demean**2).sum()
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
    slope_stderr = da.sqrt(((y_arr - y_estim)**2).sum() / dof /
                           ((x_arr - x_mean)**2).sum())
    return slope_stderr


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
    axis = cube.coord_dims(coord)[0]

    # Calculate trend
    trend_arr = da.apply_along_axis(
        _get_slope, axis, cube.data, coord.points, dtype=cube.dtype,
        shape=())
    trend_arr = da.ma.masked_invalid(trend_arr)

    # Create dummy cube by collapsing along dimension and add trend data
    aggregator = iris.analysis.Aggregator('trend', _remove_axis_np,
                                          lazy_func=_remove_axis_da)
    cube = cube.collapsed(coord, aggregator)
    cube.data = trend_arr

    # Add metadata
    if cube.var_name is not None:
        cube.var_name += '_trend'
    if cube.long_name is not None:
        cube.long_name += ' (Trend)'
    coord_units = coord.units
    if coord_units.is_time_reference():
        coord_units = Unit(coord_units.symbol.split()[0])
    invalid_units = any([cube.units is None, cube.units.is_unknown(),
                         cube.units.is_no_unit(), coord_units.is_no_unit()])
    if not invalid_units:
        cube.units /= coord_units

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
    axis = cube.coord_dims(coord)[0]

    # Calculate standard error of trends
    trend_stderr_arr = da.apply_along_axis(
        _get_slope_stderr, axis, cube.data, coord.points, dtype=cube.dtype,
        shape=())
    trend_stderr_arr = da.ma.masked_invalid(trend_stderr_arr)

    # Create dummy cube by collapsing along dimension and add trend data
    aggregator = iris.analysis.Aggregator('trend_stderr', _remove_axis_np,
                                          lazy_func=_remove_axis_da)
    cube = cube.collapsed(coord, aggregator)
    cube.data = trend_stderr_arr

    # Add metadata
    if cube.var_name is not None:
        cube.var_name += '_trend_stderr'
    if cube.long_name is not None:
        cube.long_name += ' (Trend Standard Error)'
    coord_units = coord.units
    if coord_units.is_time_reference():
        coord_units = Unit(coord_units.symbol.split()[0])
    invalid_units = any([cube.units is None, cube.units.is_unknown(),
                         cube.units.is_no_unit(), coord_units.is_no_unit()])
    if not invalid_units:
        cube.units /= coord_units

    return cube
