"""Preprocessor functions that remove trends from the data."""
import logging

import dask.array as da
import numpy as np

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
    data = da.moveaxis(da.array(np.array(cube.data)), axis, 0)
    new_shape = (data.shape[0], np.prod(data.shape[1:]))
    intercept = da.reshape(data, new_shape)
    slope = da.arange(1, data.shape[0] + 1, dtype=data.dtype) / data.shape[0]
    slope = da.stack([slope, da.ones_like(slope)], axis=1)
    slope = da.rechunk(slope)
    lstsq, _, _, _ = da.linalg.lstsq(slope, intercept)
    detrended = cube.lazy_data() - da.reshape(da.dot(slope, lstsq), cube.shape)
    return cube.copy(detrended)
