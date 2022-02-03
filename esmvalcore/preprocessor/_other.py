"""
Preprocessor functions that do not fit into any of the categories.
"""

import logging

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def clip(cube, minimum=None, maximum=None):
    """
    Clip values at a specified minimum and/or maximum value

    Values lower than minimum are set to minimum and values
    higher than maximum are set to maximum.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be clipped
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        clipped cube.
    """
    if minimum is None and maximum is None:
        raise ValueError("Either minimum, maximum or both have to be\
                          specified.")
    elif minimum is not None and maximum is not None:
        if maximum < minimum:
            raise ValueError("Maximum should be equal or larger than minimum.")
    cube.data = da.clip(cube.core_data(), minimum, maximum)
    return cube


def fix_cubes_endianness(cubes):
    """Transform cubes in big endian to little."""
    for cube in cubes:
        if cube.dtype.byteorder == ">":
            cube.data = _byteswap_array(cube.core_data())
            # Swap the coords also if neccesary
            for coord in cube.coords():
                if coord.dtype.byteorder == ">":
                    coord.points = _byteswap_array(coord.core_points())
                    if (coord.bounds is not None) and (coord.bounds.dtype.byteorder == ">"):
                        coord.bounds = _byteswap_array(coord.core_bounds())
    return cubes


def _byteswap_array(arr):
    """
    Swaps the bytes of a numpy or dask array
    """
    if isinstance(arr, da.Array):
        return _byteswap_dask_array(arr)
    elif isinstance(arr, np.ndarray):
        return _byteswap_numpy_array(arr)
    else:
        raise NotImplementedError("Data type not supported")


def _byteswap_dask_array(arr):
    """
    Swaps the bytes of a dask array

    byteswap and newbyteorder are not ufuncs and are not supported
    neither by dask or iris. The workaround is to use map_blocks
    to call the appropiate numpy functions over the dask array chunks
    returned by core_data() See
    https://github.com/dask/dask/issues/5689
    """
    swapped_da = arr.map_blocks(np.ndarray.byteswap).map_blocks(
                    np.ndarray.newbyteorder)
    return swapped_da


def _byteswap_numpy_array(arr):
    """
    Swaps the bytes of a numpy array
    """
    return arr.byteswap().newbyteorder()
