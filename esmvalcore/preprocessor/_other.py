"""Preprocessor functions that do not fit into any of the categories."""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Literal

import dask
import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError

from esmvalcore.cmor.table import get_var_info
from esmvalcore.iris_helpers import (
    ignore_iris_vague_metadata_warnings,
    rechunk_cube,
)
from esmvalcore.preprocessor._shared import (
    get_all_coord_dims,
    get_all_coords,
    get_array_module,
    get_coord_weights,
    get_weights,
    preserve_float_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from esmvalcore.cmor.table import VariableInfo

logger = logging.getLogger(__name__)


def align_metadata(
    cube: Cube,
    target_project: str,
    target_mip: str,
    target_short_name: str,
    strict: bool = True,
) -> Cube:
    """Set cube metadata to entries from a specific target project.

    This is useful to align variable metadata of different projects prior to
    performing multi-model operations (e.g.,
    :func:`~esmvalcore.preprocessor.multi_model_statistics`). For example,
    standard names differ for some variables between CMIP5 and CMIP6 which
    would prevent the calculation of multi-model statistics between CMIP5 and
    CMIP6 data.

    Parameters
    ----------
    cube:
        Input cube.
    target_project:
        Project from which target metadata is read.
    target_mip:
        MIP table from which target metadata is read.
    target_short_name:
        Variable short name from which target metadata is read.
    strict:
        If ``True``, raise an error if desired metadata cannot be read for
        variable ``target_short_name`` of MIP table ``target_mip`` and project
        ``target_project``. If ``False``, no error is raised.

    Returns
    -------
    iris.cube.Cube
        Cube with updated metadata.

    Raises
    ------
    KeyError
        Invalid ``target_project`` given.
    ValueError
        If ``strict=True``: Variable ``target_short_name`` not available for
        MIP table ``target_mip`` of project ``target_project``.

    """
    cube = cube.copy()

    try:
        var_info = _get_var_info(target_project, target_mip, target_short_name)
    except ValueError as exc:
        if strict:
            raise
        logger.debug(exc)
        return cube

    cube.long_name = var_info.long_name
    cube.standard_name = var_info.standard_name
    cube.var_name = var_info.short_name
    cube.convert_units(var_info.units)

    return cube


def _get_var_info(project: str, mip: str, short_name: str) -> VariableInfo:
    """Get variable information."""
    var_info = get_var_info(project, mip, short_name)
    if var_info is None:
        msg = (
            f"Variable '{short_name}' not available for table '{mip}' of "
            f"project '{project}'"
        )
        raise ValueError(msg)
    return var_info


def clip(
    cube: Cube,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Cube:
    """Clip values at a specified minimum and/or maximum value.

    Values lower than minimum are set to minimum and values higher than maximum
    are set to maximum.

    Parameters
    ----------
    cube:
        Input cube.
    minimum:
        Lower threshold to be applied on input cube data.
    maximum:
        Upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        Clipped cube.

    Raises
    ------
    ValueError
        ``minimum`` and ``maximum`` are set to ``None``.

    """
    if minimum is None and maximum is None:
        msg = "Either minimum, maximum or both have to be specified"
        raise ValueError(msg)
    if minimum is not None and maximum is not None:
        if maximum < minimum:
            msg = "Maximum should be equal or larger than minimum"
            raise ValueError(msg)
    cube.data = da.clip(cube.core_data(), minimum, maximum)
    return cube


@preserve_float_dtype
def cumulative_sum(
    cube: Cube,
    coord: Coord | str,
    weights: np.ndarray | da.Array | bool | None = None,
    method: Literal["sequential", "blelloch"] = "sequential",
) -> Cube:
    """Calculate cumulative sum of the elements along a given coordinate.

    Parameters
    ----------
    cube:
        Input cube.
    coord:
        Coordinate over which the cumulative sum is calculated. Must be 0D or
        1D.
    weights:
        Weights for the calculation of the cumulative sum. Each element in the
        data is multiplied by the corresponding weight before summing. Can be
        an array of the same shape as the input data, ``False`` or ``None`` (no
        weighting), or ``True`` (calculate the weights from the coordinate
        bounds; only works if each coordinate point has exactly 2 bounds).
    method:
        Method used to perform the cumulative sum. Only relevant if the cube
        has `lazy data
        <https://scitools-iris.readthedocs.io/en/stable/userguide/
        real_and_lazy_data.html>`__. See :func:`dask.array.cumsum` for details.

    Returns
    -------
    Cube
        Cube of cumulative sum. Has same dimensions and coordinates of the
        input cube.

    Raises
    ------
    iris.exceptions.CoordinateMultiDimError
        ``coord`` is not 0D or 1D.
    iris.exceptions.CoordinateNotFoundError
        ``coord`` is not found in ``cube``.

    """
    cube = cube.copy()

    # Only 0D and 1D coordinates are supported
    coord = cube.coord(coord)
    if coord.ndim > 1:
        raise CoordinateMultiDimError(coord)

    # Weighting, make sure to adapt cube standard name and units in this case
    if weights is True:
        weights = get_coord_weights(cube, coord, broadcast=True)
    if isinstance(weights, (np.ndarray, da.Array)):
        cube.data = cube.core_data() * weights
        cube.standard_name = None
        cube.units = cube.units * coord.units

    axes = get_all_coord_dims(cube, [coord])

    # For 0D coordinates, cumulative_sum is a no-op (this aligns with
    # numpy's/dask's behavior)
    if axes:
        if cube.has_lazy_data():
            cube.data = da.cumsum(
                cube.core_data(),
                axis=axes[0],
                method=method,
            )
        else:
            cube.data = np.cumsum(cube.core_data(), axis=axes[0])

    # Adapt cube metadata
    if cube.var_name is not None:
        cube.var_name = f"cumulative_{cube.var_name}"
    if cube.long_name is not None:
        cube.long_name = f"Cumulative {cube.long_name}"

    return cube


@preserve_float_dtype
def histogram(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str] | None = None,
    bins: int | Sequence[float] = 10,
    bin_range: tuple[float, float] | None = None,
    weights: np.ndarray | da.Array | bool | None = None,
    normalization: Literal["sum", "integral"] | None = None,
) -> Cube:
    """Calculate histogram.

    Very similar to :func:`numpy.histogram`, but calculates histogram only over
    the given coordinates.

    Handles lazy data and masked data.

    Parameters
    ----------
    cube:
        Input cube.
    coords:
        Coordinates over which the histogram is calculated. If ``None``,
        calculate the histogram over all coordinates, which results in a scalar
        cube.
    bins:
        If `bins` is an :obj:`int`, it defines the number of equal-width bins
        in the given `bin_range`. If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    bin_range:
        The lower and upper range of the bins. If ``None``, `bin_range` is
        simply (``cube.core_data().min(), cube.core_data().max()``). Values
        outside the range are ignored. The first element of the range must be
        less than or equal to the second. `bin_range` affects the automatic bin
        computation as well if `bins` is an :obj:`int` (see description for
        `bins` above).
    weights:
        Weights for the histogram calculation. Each value in the input data
        only contributes its associated weight towards the bin count (instead
        of 1). Weights are normalized before entering the calculation if
        `normalization` is ``'integral'`` or ``'sum'``. Can be an array of the
        same shape as the input data, ``False`` or ``None`` (no weighting), or
        ``True``. In the latter case, weighting will depend on `coords`, and
        the following coordinates will trigger weighting: `time` (will use
        lengths of time intervals as weights) and `latitude` (will use cell
        area weights). Time weights are always calculated from the input data.
        Area weights can be given as supplementary variables to the recipe
        (`areacella` or `areacello`, see :ref:`supplementary_variables`) or
        calculated from the input data (this only works for regular grids). By
        default, **NO** supplementary variables will be used; they need to be
        explicitly requested in the recipe.
    normalization:
        If ``None``, the result will contain the number of samples in each bin.
        If ``'integral'``, the result is the value of the probability `density`
        function at the bin, normalized such that the integral over the range
        is 1. If ``'sum'``, the result is the value of the probability
        `mass` function at the bin, normalized such that the sum over
        the range is 1. Normalization will be applied across `coords`, not the
        entire cube.

    Returns
    -------
    Cube
        Histogram cube. The shape of this cube will be `(x1, x2, ..., n_bins)`,
        where `xi` are the dimensions of the input cube not appearing in
        `coords` and `n_bins` is the number of bins.

    Raises
    ------
    TypeError
        Invalid `bin` type given
    ValueError
        Invalid `normalization` or `bin_range` given or `bin_range` is ``None``
        and data is fully masked.
    iris.exceptions.CoordinateNotFoundError
        A given coordinate of ``coords`` is not found in ``cube``.
    iris.exceptions.CoordinateNotFoundError
        `longitude` is not found in cube if ``weights=True``, `latitude` is in
        ``coords``, and no `cell_area` is given as
        :ref:`supplementary_variables`.

    """
    # Check arguments
    if isinstance(bins, str):
        msg = (
            f"bins cannot be a str (got '{bins}'), must be int or Sequence of "
            f"int"
        )
        raise TypeError(msg)
    allowed_norms = (None, "sum", "integral")
    if normalization is not None and normalization not in allowed_norms:
        msg = (
            f"Expected one of {allowed_norms} for normalization, got "
            f"'{normalization}'"
        )
        raise ValueError(msg)

    # If histogram is calculated over all coordinates, we can use
    # dask.array.histogram and do not need to worry about chunks; otherwise,
    # make sure that the cube is not chunked along the given coordinates
    coords = get_all_coords(cube, coords)
    axes = get_all_coord_dims(cube, coords)
    if cube.has_lazy_data() and len(axes) != cube.ndim:
        cube = rechunk_cube(cube, coords)

    # Calculate histogram
    weights = _get_histogram_weights(cube, coords, weights, normalization)
    (bin_range, bin_edges) = _get_bins(cube, bins, bin_range)
    if cube.has_lazy_data():
        func = _calculate_histogram_lazy  # type: ignore
    else:
        func = _calculate_histogram_eager  # type: ignore
    hist_data = func(
        cube.core_data(),
        weights,  # type: ignore
        along_axes=axes,
        bin_edges=bin_edges,
        bin_range=bin_range,
        normalization=normalization,
    )

    # Get final cube
    return _get_histogram_cube(
        cube,
        hist_data,
        coords,
        bin_edges,
        normalization,
    )


def _get_bins(
    cube: Cube,
    bins: int | Sequence[float],
    bin_range: tuple[float, float] | None,
) -> tuple[tuple[float, float], np.ndarray]:
    """Calculate bin range and edges."""
    if bin_range is None:
        bin_range = dask.compute(
            cube.core_data().min(),
            cube.core_data().max(),
        )
    if isinstance(bins, int):
        bin_edges = np.linspace(
            bin_range[0],
            bin_range[1],
            bins + 1,
            dtype=np.float64,
        )
    else:
        bin_edges = np.array(bins, dtype=np.float64)

    finite_bin_range = [bool(np.isfinite(r)) for r in bin_range]
    if not all(finite_bin_range):
        msg = (
            f"Cannot calculate histogram for bin_range={bin_range} (or for "
            f"fully masked data when `bin_range` is not given)"
        )
        raise ValueError(msg)

    return (bin_range, bin_edges)


def _get_histogram_weights(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
    weights: np.ndarray | da.Array | bool | None,
    normalization: Literal["sum", "integral"] | None,
) -> np.ndarray | da.Array:
    """Get histogram weights."""
    axes = get_all_coord_dims(cube, coords)
    npx = get_array_module(cube.core_data())

    weights_array: np.ndarray | da.Array
    if weights is None or weights is False:
        weights_array = npx.ones_like(cube.core_data())
    elif weights is True:
        weights_array = get_weights(cube, coords)
    else:
        weights_array = weights

    if normalization is not None:
        norm = npx.sum(weights_array, axis=axes, keepdims=True)
        weights_array = weights_array / norm

    # For lazy arrays, make sure that the chunks of the cube data and weights
    # match
    if isinstance(weights_array, da.Array):
        weights_array = weights_array.rechunk(cube.lazy_data().chunks)

    return weights_array


def _calculate_histogram_lazy(
    data: da.Array,
    weights: da.Array,
    *,
    along_axes: tuple[int, ...],
    bin_edges: np.ndarray,
    bin_range: tuple[float, float],
    normalization: Literal["sum", "integral"] | None = None,
) -> da.Array:
    """Calculate histogram over data along axes (lazy version).

    This will return an array of shape `(x1, x2, ..., n_bins)` where `xi` are
    the dimensions of `data` not appearing in `axes` and `n_bins` is the number
    of bins.

    """
    n_axes = len(along_axes)

    # (1) If histogram is calculated over all axes, use the efficient
    # da.histogram function
    if n_axes == data.ndim:
        data = data.ravel()
        weights = weights.ravel()
        mask = da.ma.getmaskarray(data)
        data = data[~mask]
        weights = weights[~mask]
        hist = da.histogram(
            data,
            bins=bin_edges,
            range=bin_range,
            weights=weights,
        )[0]
        hist_sum = hist.sum()
        hist = da.ma.masked_array(hist, mask=da.allclose(hist_sum, 0.0))
        if normalization == "sum":
            hist = hist / hist_sum
        elif normalization == "integral":
            diffs = np.array(np.diff(bin_edges), dtype=data.dtype)
            hist = hist / hist_sum / diffs
        hist = da.ma.masked_invalid(hist)

    # (2) Otherwise, use da.apply_gufunc with the eager version
    # _calculate_histogram_eager
    else:
        # da.apply_gufunc transposes the input array so that the axes given by
        # the `axes` argument to da.apply_gufunc are the rightmost dimensions.
        # Thus, we need to use `along_axes=(ndim-n_axes, ..., ndim-2, ndim-1)`
        # for _calculate_histogram_eager here.
        axes_in_chunk = tuple(range(data.ndim - n_axes, data.ndim))

        # The call signature depends also on the number of axes in `axes`, and
        # will be (a,b,...)->(nbins) where a,b,... are the data dimensions that
        # are collapsed, and nbins the number of bin centers
        in_signature = f"({','.join(list(string.ascii_lowercase)[:n_axes])})"
        hist = da.apply_gufunc(
            _calculate_histogram_eager,
            f"{in_signature},{in_signature}->(nbins)",
            data,
            weights,
            axes=[along_axes, along_axes, (-1,)],
            output_sizes={"nbins": len(bin_edges) - 1},
            along_axes=axes_in_chunk,
            bin_edges=bin_edges,
            bin_range=bin_range,
            normalization=normalization,
        )

    return hist


def _calculate_histogram_eager(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    along_axes: tuple[int, ...],
    bin_edges: np.ndarray,
    bin_range: tuple[float, float],
    normalization: Literal["sum", "integral"] | None = None,
) -> np.ndarray:
    """Calculate histogram over data along axes (eager version).

    This will return an array of shape `(x1, x2, ..., n_bins)` where `xi` are
    the dimensions of `data` not appearing in `axes` and `n_bins` is the number
    of bins.

    """
    # Create array with shape (x1, x2, ..., y) where `y` is the product of all
    # dimensions in `axes` and the `xi` are the remaining dimensions
    remaining_dims = tuple(a for a in range(data.ndim) if a not in along_axes)
    reshaped_data = np.transpose(data, axes=(*remaining_dims, *along_axes))
    reshaped_weights = np.transpose(
        weights,
        axes=(*remaining_dims, *along_axes),
    )
    shape_rem_dims = tuple(data.shape[a] for a in remaining_dims)
    reshaped_data = reshaped_data.reshape(*shape_rem_dims, -1)
    reshaped_weights = reshaped_weights.reshape(*shape_rem_dims, -1)

    # Apply vectorized version of np.histogram
    def _get_hist_values(arr, wgts):
        mask = np.ma.getmaskarray(arr)
        arr = arr[~mask]
        wgts = wgts[~mask]
        return np.histogram(
            arr,
            bins=bin_edges,
            range=bin_range,
            weights=wgts,
        )[0]

    v_histogram = np.vectorize(_get_hist_values, signature="(n),(n)->(m)")
    hist = v_histogram(reshaped_data, reshaped_weights)

    # Mask points where all input data were masked (these are the ones where
    # the histograms sums to 0)
    hist_sum = hist.sum(axis=-1, keepdims=True)
    mask = np.isclose(hist_sum, 0.0)
    hist = np.ma.array(hist, mask=np.broadcast_to(mask, hist.shape))

    # Apply normalization
    if normalization == "sum":
        hist = hist / np.ma.array(hist_sum, mask=mask)
    elif normalization == "integral":
        hist = (
            hist
            / np.ma.array(hist_sum, mask=mask)
            / np.ma.array(np.diff(bin_edges), dtype=data.dtype)
        )

    return hist


def _get_histogram_cube(
    cube: Cube,
    data: np.ndarray | da.Array,
    coords: Iterable[Coord] | Iterable[str],
    bin_edges: np.ndarray,
    normalization: Literal["sum", "integral"] | None,
):
    """Get cube with correct metadata for histogram."""
    # Calculate bin centers using 2-window running mean and get corresponding
    # coordinate
    bin_centers = np.convolve(bin_edges, np.ones(2), "valid") / 2.0
    bin_coord = DimCoord(
        bin_centers,
        bounds=np.stack((bin_edges[:-1], bin_edges[1:]), axis=-1),
        standard_name=cube.standard_name,
        long_name=cube.long_name,
        var_name=cube.var_name,
        units=cube.units,
    )

    # Get result cube with correct dimensional metadata by using dummy operation (max)
    cell_methods = cube.cell_methods
    with ignore_iris_vague_metadata_warnings():
        cube = cube.collapsed(coords, iris.analysis.MAX)

    # Get histogram cube
    long_name_suffix = (
        "" if cube.long_name is None else f" of {cube.long_name}"
    )
    var_name_suffix = "" if cube.var_name is None else f"_{cube.var_name}"
    dim_spec = [(d, cube.coord_dims(d)) for d in cube.dim_coords] + [
        (bin_coord, cube.ndim),
    ]
    if normalization == "sum":
        long_name = f"Relative Frequency{long_name_suffix}"
        var_name = f"relative_frequency{var_name_suffix}"
        units = "1"
    elif normalization == "integral":
        long_name = f"Density{long_name_suffix}"
        var_name = f"density{var_name_suffix}"
        units = cube.units**-1
    else:
        long_name = f"Frequency{long_name_suffix}"
        var_name = f"frequency{var_name_suffix}"
        units = "1"
    return Cube(
        data,
        standard_name=None,
        long_name=long_name,
        var_name=var_name,
        units=units,
        attributes=cube.attributes,
        cell_methods=cell_methods,
        dim_coords_and_dims=dim_spec,
        aux_coords_and_dims=[(a, cube.coord_dims(a)) for a in cube.aux_coords],
        aux_factories=cube.aux_factories,
        ancillary_variables_and_dims=[
            (a, cube.ancillary_variable_dims(a))
            for a in cube.ancillary_variables()
        ],
        cell_measures_and_dims=[
            (c, cube.cell_measure_dims(c)) for c in cube.cell_measures()
        ],
    )
