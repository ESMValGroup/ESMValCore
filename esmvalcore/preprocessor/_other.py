"""
Preprocessor functions that do not fit into any of the categories.
"""

import logging

import dask.array as da

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
