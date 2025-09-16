"""
Shared functions for preprocessor.

Utility functions that can be used for multiple preprocessor steps
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import CellMeasure, Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError
from iris.util import broadcast_to_shape

from esmvalcore.iris_helpers import (
    has_regular_grid,
    ignore_iris_vague_metadata_warnings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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

    # Check if valid aggregator is found
    if not hasattr(iris.analysis, cap_operator):
        msg = f"Aggregator '{operator}' not found in iris.analysis module"
        raise ValueError(
            msg,
        )
    aggregator = getattr(iris.analysis, cap_operator)
    if not hasattr(aggregator, "aggregate"):
        msg = (
            f"Aggregator {aggregator} found by '{operator}' is not a valid "
            f"iris.analysis.Aggregator"
        )
        raise ValueError(
            msg,
        )

    # Use dummy cube to check if aggregator_kwargs are valid
    x_coord = DimCoord([1.0], bounds=[0.0, 2.0], var_name="x")
    cube = Cube([0.0], dim_coords_and_dims=[(x_coord, 0)])
    test_kwargs = update_weights_kwargs(
        operator,
        aggregator,
        aggregator_kwargs,
        np.array([1.0]),
    )
    try:
        with ignore_iris_vague_metadata_warnings():
            cube.collapsed("x", aggregator, **test_kwargs)
    except (ValueError, TypeError) as exc:
        msg = f"Invalid kwargs for operator '{operator}': {exc!s}"
        raise ValueError(
            msg,
        ) from exc

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
    operator: str,
    aggregator: iris.analysis.Aggregator,
    kwargs: dict,
    weights: Any,
    cube: Cube | None = None,
    callback: Callable | None = None,
    **callback_kwargs,
) -> dict:
    """Update weights keyword argument.

    Parameters
    ----------
    operator:
        Named operator.
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
    if not aggregator_accept_weights(aggregator) and "weights" in kwargs:
        msg = f"Aggregator '{operator}' does not support 'weights' option"
        raise ValueError(
            msg,
        )
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
            normalized_cube.core_data(),
        )

    else:
        msg = (
            f"Expected 'subtract' or 'divide' for `normalize`, got "
            f"'{normalize}'"
        )
        raise ValueError(
            msg,
        )

    # Keep old metadata except for units
    new_units = normalized_cube.units
    normalized_cube.metadata = cube.metadata
    normalized_cube.units = new_units

    return normalized_cube


def _get_first_arg(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Get first argument given to a function."""
    # If positional arguments are given, use the first one
    if args:
        return args[0]

    # Otherwise, use the keyword argument given by the name of the first
    # function argument
    # Note: this function should be called AFTER func(*args, **kwargs) is run,
    # so that we can be sure that the required arguments are there
    signature = inspect.signature(func)
    first_arg_name = next(iter(signature.parameters.values())).name
    return kwargs[first_arg_name]


def preserve_float_dtype(func: Callable) -> Callable:
    """Preserve object's float dtype (all other dtypes are allowed to change).

    This can be used as a decorator for preprocessor functions to ensure that
    floating dtypes are preserved. For example, input of type float32 will
    always give output of type float32, but input of type int will be allowed
    to give output with any type.

    """
    signature = inspect.signature(func)
    if not signature.parameters:
        msg = (
            f"Cannot preserve float dtype during function '{func.__name__}', "
            f"function takes no arguments"
        )
        raise TypeError(
            msg,
        )

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> DataType:
        result = func(*args, **kwargs)
        first_arg = _get_first_arg(func, *args, **kwargs)

        if hasattr(first_arg, "dtype") and hasattr(result, "dtype"):
            dtype = first_arg.dtype
            if np.issubdtype(dtype, np.floating) and result.dtype != dtype:
                if isinstance(result, Cube):
                    result.data = result.core_data().astype(dtype)
                else:
                    result = result.astype(dtype)
        else:
            msg = (
                f"Cannot preserve float dtype during function "
                f"'{func.__name__}', the function's first argument of type "
                f"{type(first_arg)} and/or the function's return value of "
                f"type {type(result)} do not have the necessary attribute "
                f"'dtype'"
            )
            raise TypeError(
                msg,
            )

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
    coords = [c.name() if hasattr(c, "name") else c for c in coords]

    # Time weights: lengths of time interval
    if "time" in coords:
        weights = weights * get_coord_weights(cube, "time", broadcast=True)

    # Latitude weights: cell areas
    if "latitude" in coords:
        cube = cube.copy()  # avoid overwriting input cube
        if not cube.cell_measures("cell_area") and not cube.coords(
            "longitude",
        ):
            msg = (
                f"Cube {cube.summary(shorten=True)} needs a `longitude` "
                f"coordinate to calculate cell area weights (alternatively, a "
                f"`cell_area` can be given to the cube as supplementary "
                f"variable)"
            )
            raise CoordinateNotFoundError(
                msg,
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


def get_coord_weights(
    cube: Cube,
    coord: str | Coord,
    broadcast: bool = False,
) -> np.ndarray | da.core.Array:
    """Compute weighting for an arbitrary coordinate.

    Weights are calculated as the difference between the upper and lower
    bounds.

    Parameters
    ----------
    cube:
        Input cube.
    coord:
        Coordinate which is used to calculate the weights. Must have bounds
        array with 2 bounds per point.
    broadcast:
        If ``False``, weights have the shape of ``coord``. If ``True``,
        broadcast weights to shape of cube.

    Returns
    -------
    np.ndarray or da.Array
        Array of axis weights. Returns a :class:`dask.array.Array` if the input
        cube has lazy data; a :class:`numpy.ndarray` otherwise.

    """
    coord = cube.coord(coord)
    coord_dims = cube.coord_dims(coord)

    # Coordinate needs bounds of size 2
    if not coord.has_bounds():
        msg = (
            f"Cannot calculate weights for coordinate '{coord.name()}' "
            f"without bounds"
        )
        raise ValueError(
            msg,
        )
    if coord.core_bounds().shape[-1] != 2:
        msg = (
            f"Cannot calculate weights for coordinate '{coord.name()}' "
            f"with {coord.core_bounds().shape[-1]} bounds per point, expected "
            f"2 bounds per point"
        )
        raise ValueError(
            msg,
        )

    # Calculate weights of same shape as coordinate and make sure to use
    # identical chunks as parent cube for non-scalar lazy data
    weights = np.abs(coord.lazy_bounds()[:, 1] - coord.lazy_bounds()[:, 0])
    if cube.has_lazy_data() and coord_dims:
        coord_chunks = tuple(cube.lazy_data().chunks[d] for d in coord_dims)
        weights = weights.rechunk(coord_chunks)
    if not cube.has_lazy_data():
        weights = weights.compute()

    # Broadcast to cube shape if desired; scalar arrays needs special treatment
    # since iris.broadcast_to_shape cannot handle this
    if broadcast:
        chunks = cube.lazy_data().chunks if cube.has_lazy_data() else None
        if coord_dims:
            weights = broadcast_to_shape(
                weights,
                cube.shape,
                coord_dims,
                chunks=chunks,
            )
        elif cube.has_lazy_data():
            weights = da.broadcast_to(weights, cube.shape, chunks=chunks)
        else:
            weights = np.broadcast_to(weights, cube.shape)

    return weights


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
        ],
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
            msg = (
                f"If coords=None is specified, the cube "
                f"{cube.summary(shorten=True)} must not have unnamed "
                f"dimensions"
            )
            raise ValueError(
                msg,
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
    sorted_all_coord_dims = sorted(set(all_coord_dims))
    return tuple(sorted_all_coord_dims)


def _get_dims_along(cube, *args, **kwargs):
    """Get a tuple with the cube dimensions matching *args and **kwargs."""
    try:
        coord = cube.coord(*args, **kwargs, dim_coords=True)
    except iris.exceptions.CoordinateNotFoundError:
        try:
            coord = cube.coord(*args, **kwargs)
        except iris.exceptions.CoordinateNotFoundError:
            return ()
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


def apply_mask(
    mask: np.ndarray | da.Array,
    array: np.ndarray | da.Array,
    dim_map: Iterable[int],
) -> np.ma.MaskedArray | da.Array:
    """Apply a (broadcasted) mask on an array.

    Parameters
    ----------
    mask:
        The mask to apply to array.
    array:
        The array to mask out.
    dim_map :
        A mapping of the dimensions of *mask* to their corresponding
        dimension in *array*.
        See :func:`iris.util.broadcast_to_shape` for additional details.

    Returns
    -------
    np.ma.MaskedArray or da.Array:
        A copy of the input array with the mask applied.

    """
    if isinstance(array, da.Array):
        array_chunks = array.chunks
        # If the mask is not a Dask array yet, we make it into a Dask array
        # before broadcasting to avoid inserting a large array into the Dask
        # graph.
        mask_chunks = tuple(array_chunks[i] for i in dim_map)
        mask = da.asarray(mask, chunks=mask_chunks)
    else:
        array_chunks = None

    mask = iris.util.broadcast_to_shape(
        mask,
        array.shape,
        dim_map=dim_map,
        chunks=array_chunks,
    )

    array_module = get_array_module(mask, array)
    return array_module.ma.masked_where(mask, array)


def _rechunk_aux_factory_dependencies(
    cube: iris.cube.Cube,
    coord_name: str | None = None,
) -> iris.cube.Cube:
    """Rechunk coordinate aux factory dependencies.

    This ensures that the resulting coordinate has reasonably sized
    chunks that are aligned with the cube data for optimal computational
    performance.
    """
    # Workaround for https://github.com/SciTools/iris/issues/5457
    if coord_name is None:
        factories = cube.aux_factories
    else:
        try:
            factories = [cube.aux_factory(coord_name)]
        except iris.exceptions.CoordinateNotFoundError:
            return cube

    cube = cube.copy()
    cube_chunks = cube.lazy_data().chunks
    for factory in factories:
        for orig_coord in factory.dependencies.values():
            coord_dims = cube.coord_dims(orig_coord)
            if coord_dims:
                coord = orig_coord.copy()
                chunks = tuple(cube_chunks[i] for i in coord_dims)
                coord.points = coord.lazy_points().rechunk(chunks)
                if coord.has_bounds():
                    coord.bounds = coord.lazy_bounds().rechunk(
                        (*chunks, None),
                    )
                cube.replace_coord(coord)
    return cube
