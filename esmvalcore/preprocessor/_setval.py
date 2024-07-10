"""
Module containing local preprocessors for cccma standard diagnostics. 

"""

##########################################################################
# start of custom code ###################################################

import logging
import os

import cartopy.io.shapereader as shpreader
import iris
import numpy as np
import shapely.vectorized as shp_vect
from iris.analysis import Aggregator
from iris.util import rolling_window
local_prepnames = ['set_above_threshold',
                        'set_below_threshold',
                        'set_inside_range',
                        'set_outside_range']
logger = logging.getLogger(__name__)

def set_above_threshold(cube, threshold, val):
    """
    Set values above a specific threshold value to a specified value.

    Takes a value 'threshold' and sets anything that is above
    it in the cube data to val. Values equal to the threshold are not changed.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded.

    threshold: float
        threshold to be applied on input cube data.

    val: float
        value to change to 

    Returns
    -------
    iris.cube.Cube
        thresholded cube.

    """
    cube.data[cube.data > threshold] = val
    return cube


def set_below_threshold(cube, threshold, val):
    """
    Set values below a specific threshold value to a specified value.

    Takes a value 'threshold' and sets anything that is below
    it in the cube data to val. Values equal to the threshold are not changed.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    threshold: float
        threshold to be applied on input cube data.
    val: float
        value to change to 

    Returns
    -------
    iris.cube.Cube
        thresholded cube.

    """
    cube.data[cube.data > threshold] = val
    return cube


def set_inside_range(cube, minimum, maximum, val):
    """
    Set to a specified value inside a specific threshold range.

    Takes a MINIMUM and a MAXIMUM value for the range, and sets anything
    that's between the two in the cube data to a specified value.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.
    val: float
        value to change to 

    Returns
    -------
    iris.cube.Cube
        thresholded cube.

    """
    cube.data[(cube.data >= minimum) & (cube.data <= maximum)] = val
    return cube


def set_outside_range(cube, minimum, maximum, val):
    """
    Set to a specified value outside a specific threshold range.

    Takes a MINIMUM and a MAXIMUM value for the range, and sets anything
    that's outside the two, including the end points,  
    in the cube data to a specified value.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.
    val: float
        value to change to 

    Returns
    -------
    iris.cube.Cube
        thresholded cube.

    """
    cube.data[(cube.data > maximum) | (cube.data < minimum)] = val
    return cube

# end of custom code ###################################################
##########################################################################
