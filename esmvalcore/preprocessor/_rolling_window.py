"""Rolling-window operations on data cubes."""

import logging

from iris.cube import Cube

from ._shared import get_iris_aggregator, preserve_float_dtype

logger = logging.getLogger(__name__)


@preserve_float_dtype
def rolling_window_statistics(
    cube: Cube,
    coordinate: str,
    operator: str,
    window_length: int,
    **operator_kwargs,
):
    """Compute rolling-window statistics over a coordinate.

    Parameters
    ----------
    cube:
        Input cube.
    coordinate:
        Coordinate over which the rolling-window statistics is calculated.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.
    window_length:
        Size of the window to use.

    Returns
    -------
    iris.cube.Cube
        Rolling-window statistics cube.

    """
    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    cube = cube.rolling_window(coordinate, agg, window_length, *agg_kwargs)

    return cube
