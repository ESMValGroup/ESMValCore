"""
Preprocessor functions that do not fit into any of the categories.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_to_range(cube, minimum, maximum):
    """
    Set values to the given range (clipping)

    Values lower than MINIMUM are set to MINIMUM and values
    higher than MAXIMUM are set to MAXIMUM.

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
    cube.data = np.ma.clip(cube.data, minimum, maximum)
    return cube
