"""
Shared preprocessor functions.

Allows for repeated functions to be used by several preprocessor functions.
"""
import logging

import iris

logger = logging.getLogger(__name__)

allowed_operators = ['mean', 'median', 'std_dev', 'variance', 'min', 'max']
weighted_operators = ['mean',]


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
        allowed operators: mean, median, std_dev, variance, min, max
    """
    # TODO: why limit this to a small number of operations?
    operator = operator.lower()
    if operator not in allowed_operators:
        raise ValueError("operator {} not recognised. "
                         "Accepted values are: {}."
                         "".format(operator, ', '.join(allowed_operators)))
    operation = getattr(iris.analysis, operator.upper())
    return operation
