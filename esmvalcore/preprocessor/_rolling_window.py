"""Rolling-window operations on data cubes."""

import logging

from ._shared import get_iris_analysis_operation

logger = logging.getLogger(__name__)


def rolling_window_statistics(cube, coordinate, operator, window_length):
    """Compute rolling-window statistics over a coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    coordinate : str
        Coordinate over which the rolling-window statistics is calculated.
    operator : str
        Select operator to apply. Available operators: ``'mean'``,
        ``'median'``, ``'std_dev'``, ``'sum'``, ``'variance'``, ``'min'``,
        ``'max'``.
    window_length : int
        Size of the window to use.

    Returns
    -------
    iris.cube.Cube
        Rolling-window statistics cube.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError:
        Cube does not have time coordinate.
    ValueError:
        Invalid ``'operator'`` given.
    """
    operation = get_iris_analysis_operation(operator)
    # applying rolling wondow
    cube = cube.rolling_window(coordinate, operation, window_length)

    return cube
