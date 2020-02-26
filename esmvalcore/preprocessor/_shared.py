"""
Shared functions for preprocessor.

Utility functions that can be used for multiple preprocessor steps
"""
import logging

import iris
import iris.analysis

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
