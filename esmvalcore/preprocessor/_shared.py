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


def get_iris_analysis_operation(operator: str) -> iris.analysis.Aggregator:
    """Determine the iris analysis operator from a :obj:`str`.

    Map string to functional operator.

    Parameters
    ----------
    operator:
        A named operator.

    Returns
    -------
    Object that can be used within :meth:`iris.cube.Cube.collapsed`,
    :meth:`iris.cube.Cube.aggregated_by`, or
    :meth:`iris.cube.Cube.rolling_window`.

    Raises
    ------
    ValueError
        An invalid operator is specified. Allowed options: `mean`, `median`,
        `std_dev`, `sum`, `variance`, `min`, `max`, `rms`.
    """
    operators = [
        'mean', 'median', 'std_dev', 'sum', 'variance', 'min', 'max', 'rms',
    ]
    operator = operator.lower()
    if operator not in operators:
        raise ValueError(
            f"operator '{operator}' not recognised. Accepted values are: "
            f"{', '.join(operators)}."
        )
    operation = getattr(iris.analysis, operator.upper())
    return operation


def operator_accept_weights(operator: str) -> bool:
    """Get if operator support weights.

    Parameters
    ----------
    operator:
        A named operator.

    Returns
    -------
    ``True`` if operator support weights, ``False`` otherwise.

    """
    return operator.lower() in ('mean', 'sum', 'rms')
