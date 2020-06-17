"""Preprocessor functions that remove trends from the data."""
import logging

import dask.array as da
import scipy.signal

logger = logging.getLogger(__name__)


def detrend(cube, dimension='time', method='linear'):
    """
    Detrend data along a given dimension.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        scipy.signal.detrend,
        axis=axis,
        arr=cube.lazy_data(),
        type=method,
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)


def linear_trend(cube, dimension='time'):
    """
    Give the trend along a given dimension by detrending first and returning
    the difference

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    # from cf_units import Unit

    cube_detrend = detrend(cube, dimension=dimension, method='linear')
    cube.data -= cube_detrend.data

    # # if frequency not in ['yr', 'mon', 'day']
    # if frequency == 'mon':
    #     frequency = 'month'
    # cube.units = Unit(str(cube.units) + f' {frequency}-1')

    cube.long_name = cube.long_name + " trend"

    return cube