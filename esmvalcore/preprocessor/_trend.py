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


def _get_slope(y_arr, x_arr):
    """Calculate slope between X and Y array."""
    if np.ma.is_masked(y_arr):
        x_arr = x_arr[~y_arr.mask]
        y_arr = y_arr[~y_arr.mask]

    # If less than 2 points, slope calculation not possible
    if len(y_arr) < 2:
        return np.nan

    # Calculate slope
    x_demean = x_arr - x_arr.mean()
    y_demean = y_arr - y_arr.mean()
    slope = da.sum(x_demean * y_demean) / da.sum(x_demean**2)
    return slope


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
        _get_slope, axis, cube.core_data(), coord.core_points(),
        dtype=cube.dtype, shape=())
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
    invalid_units = np.any([cube.units is None, cube.units.is_unknown(),
                            cube.units.is_no_unit(), coord_units.is_no_unit()])
    if not invalid_units:
        cube.units /= coord_units

    return cube
