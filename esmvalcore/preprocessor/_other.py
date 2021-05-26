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


def fix_cube_endianess(cubes):
    """Transform cubes in big endian to little."""
    for cube in cubes:
        if cube.dtype.byteorder == ">":
            if cube.has_lazy_data():
                # byteswap and newbyteorder are not ufuncs and are not supported
                # neither by dask or iris. The workaround is to use map_blocks
                # to call the appropiate numpy functions over the dask array
                # returned by core_data() See
                # https://github.com/dask/dask/issues/5689
                cube.data = cube.core_data().map_blocks(
                    np.ndarray.byteswap, True).map_blocks(
                    np.ndarray.newbyteorder
                )
                cube.coords =
            else:
                # Directly swap the data
                cube.data = cube.data.byteswap().newbyteorder()
    return cubes
