"""
Shared functions for preprocessor.

Utility functions that can be used for multiple preprocessor steps
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Literal, Optional

import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import CellMeasure, Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError
from iris.util import broadcast_to_shape

from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.iris_helpers import has_regular_grid
from esmvalcore.typing import DataType

logger = logging.getLogger(__name__)


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
    if cap_operator == "STD":
        msg = (
            f"The operator '{operator}' for computing the standard deviation "
            f"has been deprecated in ESMValCore version 2.10.0 and is "
            f"scheduled for removal in version 2.12.0. Please use 'std_dev' "
            f"instead. This is an exact replacement."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        operator = "std_dev"
        cap_operator = "STD_DEV"
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
        aggregator_kwargs["percent"] = float(operator[1:])
        operator = "percentile"
        cap_operator = "PERCENTILE"

    # Check if valid aggregator is found
    if not hasattr(iris.analysis, cap_operator):
        raise ValueError(
            f"Aggregator '{operator}' not found in iris.analysis module"
        )
    aggregator = getattr(iris.analysis, cap_operator)
    if not hasattr(aggregator, "aggregate"):
        raise ValueError(
            f"Aggregator {aggregator} found by '{operator}' is not a valid "
            f"iris.analysis.Aggregator"
        )

    # Use dummy cube to check if aggregator_kwargs are valid
    x_coord = DimCoord([1.0], bounds=[0.0, 2.0], var_name="x")
    cube = Cube([0.0], dim_coords_and_dims=[(x_coord, 0)])
    test_kwargs = update_weights_kwargs(
        aggregator, aggregator_kwargs, np.array([1.0])
    )
    try:
        cube.collapsed("x", aggregator, **test_kwargs)
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
    if aggregator_accept_weights(aggregator) and kwargs.get("weights", True):
        kwargs["weights"] = weights
        if cube is not None and callback is not None:
            callback(cube, **callback_kwargs)
    else:
        kwargs.pop("weights", None)
    return kwargs


def get_normalized_cube(
    cube: Cube,
    statistics_cube: Cube,
    normalize: Literal["subtract", "divide"],
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
    if normalize == "subtract":
        normalized_cube = cube - statistics_cube

    elif normalize == "divide":
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


def preserve_float_dtype(func: Callable) -> Callable:
    """Preserve object's float dtype (all other dtypes are allowed to change).

    This can be used as a decorator for preprocessor functions to ensure that
    floating dtypes are preserved. For example, input of type float32 will
    always give output of type float32, but input of type int will be allowed
    to give output with any type.

    """

    @wraps(func)
    def wrapper(data: DataType, *args: Any, **kwargs: Any) -> DataType:
        dtype = data.dtype
        result = func(data, *args, **kwargs)
        if np.issubdtype(dtype, np.floating) and result.dtype != dtype:
            if isinstance(result, Cube):
                result.data = result.core_data().astype(dtype)
            else:
                result = result.astype(dtype)
        return result

    return wrapper


def _groupby(iterable, keyfunc):
    """Group iterable by key function.

    The items are grouped by the value that is returned by the `keyfunc`

    Parameters
    ----------
    iterable : list, tuple or iterable
        List of items to group
    keyfunc : callable
        Used to determine the group of each item. These become the keys
        of the returned dictionary

    Returns
    -------
    dict
        Returns a dictionary with the grouped values.
    """
    grouped = defaultdict(set)
    for item in iterable:
        key = keyfunc(item)
        grouped[key].add(item)

    return grouped


def _group_products(products, by_key):
    """Group products by the given list of attributes."""

    def grouper(product):
        return product.group(by_key)

    grouped = _groupby(products, keyfunc=grouper)
    return grouped.items()


def get_array_module(*args):
    """Return the best matching array module.

    If at least one of the arguments is a :class:`dask.array.Array` object,
    the :mod:`dask.array` module is returned. In all other cases the
    :mod:`numpy` module is returned.
    """
    for arg in args:
        if isinstance(arg, da.Array):
            return da
    return np


def get_weights(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
) -> np.ndarray | da.Array:
    """Calculate suitable weights for given coordinates."""
    npx = get_array_module(cube.core_data())
    weights = npx.ones_like(cube.core_data())

    # Time weights: lengths of time interval
    if "time" in coords:
        weights = weights * broadcast_to_shape(
            npx.array(get_time_weights(cube)),
            cube.shape,
            cube.coord_dims("time"),
            chunks=cube.lazy_data().chunks if cube.has_lazy_data() else None,
        )

    # Latitude weights: cell areas
    if "latitude" in coords:
        cube = cube.copy()  # avoid overwriting input cube
        if not cube.cell_measures("cell_area") and not cube.coords(
            "longitude"
        ):
            raise CoordinateNotFoundError(
                f"Cube {cube.summary(shorten=True)} needs a `longitude` "
                f"coordinate to calculate cell area weights for weighted "
                f"distance metric over coordinates {coords} (alternatively, "
                f"a `cell_area` can be given to the cube as supplementary "
                f"variable)"
            )
        try_adding_calculated_cell_area(cube)
        area_weights = cube.cell_measure("cell_area").core_data()
        if cube.has_lazy_data():
            area_weights = da.array(area_weights)
            chunks = cube.lazy_data().chunks
        else:
            chunks = None
        weights = weights * broadcast_to_shape(
            area_weights,
            cube.shape,
            cube.cell_measure_dims("cell_area"),
            chunks=chunks,
        )

    return weights


def get_time_weights(cube: Cube) -> np.ndarray | da.core.Array:
    """Compute the weighting of the time axis.

    Parameters
    ----------
    cube:
        Input cube.

    Returns
    -------
    np.ndarray or da.Array
        Array of time weights for averaging. Returns a
        :class:`dask.array.Array` if the input cube has lazy data; a
        :class:`numpy.ndarray` otherwise.

    """
    time = cube.coord("time")
    coord_dims = cube.coord_dims("time")

    # Multidimensional time coordinates are not supported: In this case,
    # weights cannot be simply calculated as difference between the bounds
    if len(coord_dims) > 1:
        raise ValueError(
            f"Weighted statistical operations are not supported for "
            f"{len(coord_dims):d}D time coordinates, expected 0D or 1D"
        )

    # Extract 1D time weights (= lengths of time intervals)
    time_weights = time.lazy_bounds()[:, 1] - time.lazy_bounds()[:, 0]
    if cube.has_lazy_data():
        # Align the weight chunks with the data chunks to avoid excessively
        # large chunks as a result of broadcasting.
        time_chunks = cube.lazy_data().chunks[coord_dims[0]]
        time_weights = time_weights.rechunk(time_chunks)
    else:
        time_weights = time_weights.compute()
    return time_weights


def try_adding_calculated_cell_area(cube: Cube) -> None:
    """Try to add calculated cell measure 'cell_area' to cube (in-place)."""
    if cube.cell_measures("cell_area"):
        return

    logger.debug(
        "Found no cell measure 'cell_area' in cube %s. Check availability of "
        "supplementary variables",
        cube.summary(shorten=True),
    )
    logger.debug("Attempting to calculate grid cell area")

    rotated_pole_grid = all(
        [
            cube.coord("latitude").core_points().ndim == 2,
            cube.coord("longitude").core_points().ndim == 2,
            cube.coords("grid_latitude"),
            cube.coords("grid_longitude"),
        ]
    )

    # For regular grids, calculate grid cell areas with iris function
    if has_regular_grid(cube):
        cube = guess_bounds(cube, ["latitude", "longitude"])
        logger.debug("Calculating grid cell areas for regular grid")
        cell_areas = _compute_area_weights(cube)

    # For rotated pole grids, use grid_latitude and grid_longitude to calculate
    # grid cell areas
    elif rotated_pole_grid:
        cube = guess_bounds(cube, ["grid_latitude", "grid_longitude"])
        cube_tmp = cube.copy()
        cube_tmp.remove_coord("latitude")
        cube_tmp.coord("grid_latitude").rename("latitude")
        cube_tmp.remove_coord("longitude")
        cube_tmp.coord("grid_longitude").rename("longitude")
        logger.debug("Calculating grid cell areas for rotated pole grid")
        cell_areas = _compute_area_weights(cube_tmp)

    # For all other cases, grid cell areas cannot be calculated
    else:
        logger.error(
            "Supplementary variables are needed to calculate grid cell "
            "areas for irregular or unstructured grid of cube %s",
            cube.summary(shorten=True),
        )
        raise CoordinateMultiDimError(cube.coord("latitude"))

    # Add new cell measure
    cell_measure = CellMeasure(
        cell_areas,
        standard_name="cell_area",
        units="m2",
        measure="area",
    )
    cube.add_cell_measure(cell_measure, np.arange(cube.ndim))


def _compute_area_weights(cube):
    """Compute area weights."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings(
            "always",
            message="Using DEFAULT_SPHERICAL_EARTH_RADIUS.",
            category=UserWarning,
            module="iris.analysis.cartography",
        )
        if cube.has_lazy_data():
            kwargs = {"compute": False, "chunks": cube.lazy_data().chunks}
        else:
            kwargs = {"compute": True}
        weights = iris.analysis.cartography.area_weights(cube, **kwargs)
        for warning in caught_warnings:
            logger.debug(
                "%s while computing area weights of the following cube:\n%s",
                warning.message,
                cube,
            )
    return weights


def get_all_coords(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str] | None,
) -> Iterable[Coord] | Iterable[str]:
    """Get all desired coordinates in a cube."""
    if coords is None:
        coords = [c.name() for c in cube.dim_coords]
        if len(coords) != cube.ndim:
            raise ValueError(
                f"If coords=None is specified, the cube "
                f"{cube.summary(shorten=True)} must not have unnamed "
                f"dimensions"
            )
    return coords


def get_all_coord_dims(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
) -> tuple[int, ...]:
    """Get sorted list of all coordinate dimensions from coordinates."""
    all_coord_dims = []
    for coord in coords:
        all_coord_dims.extend(cube.coord_dims(coord))
    sorted_all_coord_dims = sorted(list(set(all_coord_dims)))
    return tuple(sorted_all_coord_dims)


def _get_dims_along(cube, *args, **kwargs):
    """Get a tuple with the cube dimensions matching *args and **kwargs."""
    try:
        coord = cube.coord(*args, **kwargs, dim_coords=True)
    except iris.exceptions.CoordinateNotFoundError:
        try:
            coord = cube.coord(*args, **kwargs)
        except iris.exceptions.CoordinateNotFoundError:
            return tuple()
    return cube.coord_dims(coord)


def get_dims_along_axes(
    cube: iris.cube.Cube,
    axes: Iterable[Literal["T", "Z", "Y", "X"]],
) -> tuple[int, ...]:
    """Get a tuple with the dimensions along one or more axis."""
    dims = {d for axis in axes for d in _get_dims_along(cube, axis=axis)}
    return tuple(sorted(dims))


def get_dims_along_coords(
    cube: iris.cube.Cube,
    coords: Iterable[str],
) -> tuple[int, ...]:
    """Get a tuple with the dimensions along one or more coordinates."""
    dims = {d for coord in coords for d in _get_dims_along(cube, coord)}
    return tuple(sorted(dims))
