"""Preprocessor functions that remove trends from the data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import dask.array as da
import scipy.signal

if TYPE_CHECKING:
    from iris.cube import Cube

logger = logging.getLogger(__name__)


def detrend(
    cube: Cube,
    dimension: str = "time",
    method: Literal["linear", "constant"] = "linear",
) -> Cube:
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
        shape=(cube.shape[axis],),
    )
    return cube.copy(detrended)
