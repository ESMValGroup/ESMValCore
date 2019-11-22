"""
Shared functions for preprocessor.

Utility functions that can be used for multiple preprocessor steps
"""
import logging

import iris
import iris.analysis
import numpy as np
from cf_units import Unit

logger = logging.getLogger(__name__)


# guess bounds tool
def guess_bounds(cube, coords):
    """Guess bounds of a cube, or not."""
    # check for bounds just in case
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    return cube


def get_iris_analysis_operation(operator):
    """
    Determine the iris analysis operator from a string.

    Map string to functional operator.

    Parameters
    ----------
    operator: str
        A named operator.

    Returns
    -------
        function: A function from iris.analysis

    Raises
    ------
    ValueError
        operator not in allowed operators list.
        allowed operators: mean, median, std_dev, sum, variance, min, max
    """
    operators = ['mean', 'median', 'std_dev', 'sum', 'variance', 'min', 'max']
    operator = operator.lower()
    if operator not in operators:
        raise ValueError("operator {} not recognised. "
                         "Accepted values are: {}."
                         "".format(operator, ', '.join(operators)))
    operation = getattr(iris.analysis, operator.upper())
    return operation


def operator_accept_weights(operator):
    """
    Get if operator support weights.

    Parameters
    ----------
    operator: str
        A named operator.

    Returns
    -------
        bool: True if operator support weights, False otherwise

    """
    return operator.lower() in ('mean', 'sum')


def make_1dim_coords(cube):
    """
    Make lon/lat AuxCoords 1-dim for ocean data.

    This function must be used before running
    iris.analysis.cartography.area_weights().
    """
    # Process lat and lon if they are 2dim
    if cube.coord("latitude").ndim > 1 and cube.coord("longitude").ndim > 1:
        # get coordinate positional index
        if len(cube.dim_coords) == 4:
            lat_idx = 2
            lon_idx = 3
        elif len(cube.dim_coords) == 3:
            lat_idx = 1
            lon_idx = 2
        elif len(cube.dim_coords) == 2:
            lat_idx = 0
            lon_idx = 1

        # sort points and assign 1-dim arrays
        lat_points = np.sort(cube.coord("latitude").points[:, 0])
        lon_points = np.sort(cube.coord("longitude").points[0, :])
        cube.remove_coord("latitude")
        cube.remove_coord("longitude")
        cube.add_aux_coord(
            iris.coords.AuxCoord(lat_points,
                                 standard_name='latitude',
                                 units=Unit('degrees')), lat_idx)
        cube.add_aux_coord(
            iris.coords.AuxCoord(lon_points,
                                 standard_name='longitude',
                                 units=Unit('degrees')), lon_idx)
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

    return cube
