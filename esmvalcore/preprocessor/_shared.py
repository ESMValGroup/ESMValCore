"""
Shared functions for preprocessor.

Utility functions that can be used for multiple preprocessor steps
"""
from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from typing import Any, Literal, Optional

import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import DTypeLike

from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.typing import DataType

logger = logging.getLogger(__name__)


# guess bounds tool
def guess_bounds(cube, coords):
    """Guess bounds of a cube, or not."""
    # check for bounds just in case
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    return cube


def get_iris_aggregator(
    operator: str,
    **operator_kwargs,
) -> tuple[iris.analysis.Aggregator, dict]:
    """Get :class:`iris.analysis.Aggregator` and keyword arguments.

    Supports all available aggregators in :mod:`iris.analysis`.

    Parameters
    ----------
    operator:
        A named operator that is used to search for aggregators. Will be
        capitalized before searching for aggregators, i.e., `MEAN` **and**
        `mean` will find :const:`iris.analysis.MEAN`.
    **operator_kwargs:
        Optional keyword arguments for the aggregator.

    Returns
    -------
    tuple[iris.analysis.Aggregator, dict]
        Object that can be used within :meth:`iris.cube.Cube.collapsed`,
        :meth:`iris.cube.Cube.aggregated_by`, or
        :meth:`iris.cube.Cube.rolling_window` and the corresponding keyword
        arguments.

    Raises
    ------
    ValueError
        An invalid `operator` is specified, i.e., it is not found in
        :mod:`iris.analysis` or the returned object is not an
        :class:`iris.analysis.Aggregator`.

    """
    cap_operator = operator.upper()
    aggregator_kwargs = dict(operator_kwargs)

    # Deprecations
    if cap_operator == 'STD':
        msg = (
            f"The operator '{operator}' for computing the standard deviation "
            f"has been deprecated in ESMValCore version 2.10.0 and is "
            f"scheduled for removal in version 2.12.0. Please use 'std_dev' "
            f"instead. This is an exact replacement."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        operator = 'std_dev'
        cap_operator = 'STD_DEV'
    elif re.match(r"^(P\d{1,2})(\.\d*)?$", cap_operator):
        msg = (
            f"Specifying percentile operators with the syntax 'pXX.YY' (here: "
            f"'{operator}') has been deprecated in ESMValCore version 2.10.0 "
            f"and is scheduled for removal in version 2.12.0. Please use "
            f"`operator='percentile'` with the keyword argument "
            f"`percent=XX.YY` instead. Example: `percent=95.0` for 'p95.0'. "
            f"This is an exact replacement."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        aggregator_kwargs['percent'] = float(operator[1:])
        operator = 'percentile'
        cap_operator = 'PERCENTILE'

    # Check if valid aggregator is found
    if not hasattr(iris.analysis, cap_operator):
        raise ValueError(
            f"Aggregator '{operator}' not found in iris.analysis module"
        )
    aggregator = getattr(iris.analysis, cap_operator)
    if not hasattr(aggregator, 'aggregate'):
        raise ValueError(
            f"Aggregator {aggregator} found by '{operator}' is not a valid "
            f"iris.analysis.Aggregator"
        )

    # Use dummy cube to check if aggregator_kwargs are valid
    x_coord = DimCoord([1.0], bounds=[0.0, 2.0], var_name='x')
    cube = Cube([0.0], dim_coords_and_dims=[(x_coord, 0)])
    test_kwargs = update_weights_kwargs(
        aggregator, aggregator_kwargs, np.array([1.0])
    )
    try:
        cube.collapsed('x', aggregator, **test_kwargs)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid kwargs for operator '{operator}': {str(exc)}"
        )

    return (aggregator, aggregator_kwargs)


def aggregator_accept_weights(aggregator: iris.analysis.Aggregator) -> bool:
    """Check if aggregator support weights.

    Parameters
    ----------
    aggregator:
        Aggregator to check.

    Returns
    -------
    bool
        ``True`` if aggregator support weights, ``False`` otherwise.

    """
    weighted_aggregators_cls = (
        iris.analysis.WeightedAggregator,
        iris.analysis.WeightedPercentileAggregator,
    )
    return isinstance(aggregator, weighted_aggregators_cls)


def update_weights_kwargs(
    aggregator: iris.analysis.Aggregator,
    kwargs: dict,
    weights: Any,
    cube: Optional[Cube] = None,
    callback: Optional[Callable] = None,
    **callback_kwargs,
) -> dict:
    """Update weights keyword argument.

    Parameters
    ----------
    aggregator:
        Iris aggregator.
    kwargs:
        Keyword arguments to update.
    weights:
        Object which will be used as weights if supported and desired.
    cube:
        Cube which can be updated through the callback (if not None) if weights
        are used.
    callback:
        Optional callback function with signature `f(cube: iris.cube.Cube,
        **kwargs) -> None`. Should update the cube given to this function
        in-place. Is called only when weights are used and cube is not None.
    **callback_kwargs:
        Optional keyword arguments passed to `callback`.

    Returns
    -------
    dict
        Updated keyword arguments.

    """
    kwargs = dict(kwargs)
    if aggregator_accept_weights(aggregator) and kwargs.get('weights', True):
        kwargs['weights'] = weights
        if cube is not None and callback is not None:
            callback(cube, **callback_kwargs)
    else:
        kwargs.pop('weights', None)
    return kwargs


def get_normalized_cube(
    cube: Cube,
    statistics_cube: Cube,
    normalize: Literal['subtract', 'divide'],
) -> Cube:
    """Get cube normalized with statistics cube.

    Parameters
    ----------
    cube:
        Input cube that will be normalized.
    statistics_cube:
        Cube that is used to normalize the input cube. Needs to be
        broadcastable to the input cube's shape according to iris' rich
        broadcasting rules enabled by the use of named dimensions (see also
        https://scitools-iris.readthedocs.io/en/latest/userguide/cube_maths.
        html#calculating-a-cube-anomaly). This is usually ensure by using
        :meth:`iris.cube.Cube.collapsed` to calculate the statistics cube.
    normalize:
        Normalization operation. Can either be `subtract` (statistics cube is
        subtracted from the input cube) or `divide` (input cube is divided by
        the statistics cube).

    Returns
    -------
    Cube
        Input cube normalized with statistics cube.

    """
    if normalize == 'subtract':
        normalized_cube = cube - statistics_cube

    elif normalize == 'divide':
        normalized_cube = cube / statistics_cube

        # Iris sometimes masks zero-divisions, sometimes not
        # (https://github.com/SciTools/iris/issues/5523). Make sure to
        # consistently mask them here.
        normalized_cube.data = da.ma.masked_invalid(
            normalized_cube.core_data()
        )

    else:
        raise ValueError(
            f"Expected 'subtract' or 'divide' for `normalize`, got "
            f"'{normalize}'"
        )

    # Keep old metadata except for units
    new_units = normalized_cube.units
    normalized_cube.metadata = cube.metadata
    normalized_cube.units = new_units

    return normalized_cube


def adapt_float_dtype(data: DataType, dtype: DTypeLike) -> DataType:
    """Adapt float dtype of data (all other dtypes are left unchanged)."""
    if np.issubdtype(dtype, np.floating) and data.dtype != dtype:
        if isinstance(data, Cube):
            data.data = data.core_data().astype(dtype)
        else:
            data = data.astype(dtype)
    return data


def preserve_float_dtype(func: Callable) -> Callable:
    """Preserve object's float dtype (all other dtypes are allowed to change).

    This can be used as a decorator for preprocessor functions to ensure that
    floating dtypes are preserved. For example, input of type float32 will
    always give output of type float32, but input of type int will be allowed
    to given output with any type.

    """

    def inner(data: DataType, *args: Any, **kwargs: Any) -> DataType:
        dtype = data.dtype
        result = func(data, *args, **kwargs)
        result = adapt_float_dtype(result, dtype)
        return result

    return inner
