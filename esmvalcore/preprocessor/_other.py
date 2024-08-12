"""Preprocessor functions that do not fit into any of the categories."""
from __future__ import annotations

import logging
import string
from collections.abc import Iterable, Sequence
from typing import Literal

import dask
import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import Coord, DimCoord
from iris.cube import Cube

from esmvalcore.iris_helpers import rechunk_cube
from esmvalcore.preprocessor._shared import (
    get_all_coord_dims,
    get_all_coords,
    get_array_module,
    get_weights,
    preserve_float_dtype,
)

logger = logging.getLogger(__name__)


def clip(cube, minimum=None, maximum=None):
    """Clip values at a specified minimum and/or maximum value.

    Values lower than minimum are set to minimum and values
    higher than maximum are set to maximum.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be clipped
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        clipped cube.
    """
    if minimum is None and maximum is None:
        raise ValueError("Either minimum, maximum or both have to be\
                          specified.")
    elif minimum is not None and maximum is not None:
        if maximum < minimum:
            raise ValueError("Maximum should be equal or larger than minimum.")
    cube.data = da.clip(cube.core_data(), minimum, maximum)
    return cube


@preserve_float_dtype
def histogram(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str] | None = None,
    bins: int | Sequence[float] = 10,
    bin_range: tuple[float, float] | None = None,
    weights: np.ndarray | da.Array | bool | None = None,
    normalization: Literal['sum', 'integral'] | None = None,
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
        `longitude` is not found in cube if `weights=True`, `latitude` is in
        `coords`, and no `cell_area` is given as
        :ref:`supplementary_variables`.

    """
    # Check arguments
    if isinstance(bins, str):
        raise TypeError(
            f"bins cannot be a str (got '{bins}'), must be int or Sequence of "
            f"int"
        )
    allowed_norms = (None, 'sum', 'integral')
    if normalization is not None and normalization not in allowed_norms:
        raise ValueError(
            f"Expected one of {allowed_norms} for normalization, got "
            f"'{normalization}'"
        )

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
    hist_cube = _get_histogram_cube(
        cube, hist_data, coords, bin_edges, normalization
    )

    return hist_cube


def _get_bins(
    cube: Cube,
    bins: int | Sequence[float],
    bin_range: tuple[float, float] | None,
) -> tuple[tuple[float, float], np.ndarray]:
    """Calculate bin range and edges."""
    if bin_range is None:
        bin_range = dask.compute(
            cube.core_data().min(), cube.core_data().max()
        )
    if isinstance(bins, int):
        bin_edges = np.linspace(
            bin_range[0], bin_range[1], bins + 1, dtype=np.float64
        )
    else:
        bin_edges = np.array(bins, dtype=np.float64)

    finite_bin_range = [bool(np.isfinite(r)) for r in bin_range]
    if not all(finite_bin_range):
        raise ValueError(
            f"Cannot calculate histogram for bin_range={bin_range} (or for "
            f"fully masked data when `bin_range` is not given)"
        )

    return (bin_range, bin_edges)


def _get_histogram_weights(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
    weights: np.ndarray | da.Array | bool | None,
    normalization: Literal['sum', 'integral'] | None,
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
    normalization: Literal['sum', 'integral'] | None = None,
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
            data, bins=bin_edges, range=bin_range, weights=weights
        )[0]
        hist_sum = hist.sum()
        hist = da.ma.masked_array(hist, mask=da.allclose(hist_sum, 0.0))
        if normalization == 'sum':
            hist = hist / hist_sum
        elif normalization == 'integral':
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
        axes_in_chunk = tuple(range(data.ndim - n_axes,  data.ndim))

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
            output_sizes={'nbins': len(bin_edges) - 1},
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
    normalization: Literal['sum', 'integral'] | None = None,
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
        weights, axes=(*remaining_dims, *along_axes)
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
            arr, bins=bin_edges, range=bin_range, weights=wgts
        )[0]

    v_histogram = np.vectorize(_get_hist_values, signature='(n),(n)->(m)')
    hist = v_histogram(reshaped_data, reshaped_weights)

    # Mask points where all input data were masked (these are the ones where
    # the histograms sums to 0)
    hist_sum = hist.sum(axis=-1, keepdims=True)
    mask = np.isclose(hist_sum, 0.0)
    hist = np.ma.array(hist, mask=np.broadcast_to(mask, hist.shape))

    # Apply normalization
    if normalization == 'sum':
        hist = hist / np.ma.array(hist_sum, mask=mask)
    elif normalization == 'integral':
        hist = (
            hist /
            np.ma.array(hist_sum, mask=mask) /
            np.ma.array(np.diff(bin_edges), dtype=data.dtype)
        )

    return hist


def _get_histogram_cube(
    cube: Cube,
    data: np.ndarray | da.Array,
    coords: Iterable[Coord] | Iterable[str],
    bin_edges: np.ndarray,
    normalization: Literal['sum', 'integral'] | None,
):
    """Get cube with correct metadata for histogram."""
    # Calculate bin centers using 2-window running mean and get corresponding
    # coordinate
    bin_centers = np.convolve(bin_edges, np.ones(2), 'valid') / 2.0
    bin_coord = DimCoord(
        bin_centers,
        bounds=np.stack((bin_edges[:-1], bin_edges[1:]), axis=-1),
        standard_name=cube.standard_name,
        long_name=cube.long_name,
        var_name=cube.var_name,
        units=cube.units,
    )

    # Get result cube with correct dimensional metadata by using dummy
    # operation (max)
    cell_methods = cube.cell_methods
    cube = cube.collapsed(coords, iris.analysis.MAX)

    # Get histogram cube
    long_name_suffix = (
        '' if cube.long_name is None else f' of {cube.long_name}'
    )
    var_name_suffix = '' if cube.var_name is None else f'_{cube.var_name}'
    dim_spec = (
        [(d, cube.coord_dims(d)) for d in cube.dim_coords] +
        [(bin_coord, cube.ndim)]
    )
    if normalization == 'sum':
        long_name = f"Relative Frequency{long_name_suffix}"
        var_name = f"relative_frequency{var_name_suffix}"
        units = '1'
    elif normalization == 'integral':
        long_name = f"Density{long_name_suffix}"
        var_name = f"density{var_name_suffix}"
        units = cube.units**-1
    else:
        long_name = f"Frequency{long_name_suffix}"
        var_name = f"frequency{var_name_suffix}"
        units = '1'
    hist_cube = Cube(
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
            (a, cube.ancillary_variable_dims(a)) for a in
            cube.ancillary_variables()
        ],
        cell_measures_and_dims=[
            (c, cube.cell_measure_dims(c)) for c in cube.cell_measures()
        ],
    )

    return hist_cube
