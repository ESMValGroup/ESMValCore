"""
Preprocessor functions that do not fit into any of the categories.
"""

import logging
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
    cube.data = np.ma.clip(cube.data, minimum, maximum)
    return cube
