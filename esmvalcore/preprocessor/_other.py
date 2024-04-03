"""Preprocessor functions that do not fit into any of the categories."""
from __future__ import annotations

import logging
import string
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

import dask
import dask.array as da
import iris.analysis
import numpy as np
from iris.coords import CellMethod, Coord, DimCoord
from iris.cube import Cube

from esmvalcore.iris_helpers import add_leading_dim_to_cube, rechunk_cube

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


def histogram(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str] | None = None,
    bins: int | Sequence[float] = 10,
    bin_range: tuple[float, float] | None = None,
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
        The lower and upper range of the bins. If not provided, `bin_range` is
        simply (``cube.core_data().min(), cube.core_data().max()``). Values
        outside the range are ignored. The first element of the range must be
        less than or equal to the second. `bin_range` affects the automatic bin
        computation as well if `bins` is an :obj:`int` (see description for
        `bins` above).
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
        Invalid `normalization` given.

    """
    # Check arguments
    if isinstance(bins, str):
        raise TypeError("bins cannot be a str, must be int or Sequence of int")
    allowed_norms = (None, 'sum', 'integral')
    if normalization is not None and normalization not in allowed_norms:
        raise ValueError(
            f"Expected one of {allowed_norms} for normalization, got "
            f"'{normalization}'"
        )

    # Calculate bin edges
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

    # If histogram is calculated over all coordinates, we can use
    # dask.array.histogram and do not need to worry about chunks; otherwise,
    # make sure that the cube is not chunked along the given coordinates
    coords = get_all_coords(cube, coords)
    axes = get_all_coord_dims(cube, coords)
    if cube.has_lazy_data() and len(axes) == cube.ndim:
        cube = rechunk_cube(cube, coords)

    # Calculate histogram
    if cube.has_lazy_data():
        func = _calculate_histogram_lazy  # type: ignore
    else:
        func = _calculate_histogram_eager  # type: ignore
    hist_data = func(
        cube.core_data(),
        along_axes=axes,
        bin_edges=bin_edges,
        bin_range=bin_range,
        normalization=normalization,
    )

    # Get final cube with correct metadata and data
    hist_cube = _get_histogram_cube(cube, coords, bin_edges, normalization)
    hist_cube.data = hist_data.astype(cube.dtype)

    return hist_cube


def _calculate_histogram_lazy(
    data: da.Array,
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

    # If histogram is calculated over all axes, use the efficient da.histogram
    # function
    if n_axes == data.ndim:
        data = data.ravel()
        data = data[~da.ma.getmaskarray(data)]
        hist = da.histogram(data, bins=bin_edges, range=bin_range)[0]
        if normalization == 'sum':
            hist = hist / hist.sum()
        elif normalization == 'integral':
            diffs = np.array(np.diff(bin_edges), dtype=data.dtype)
            hist = hist / hist.sum() / diffs
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
        hist = da.apply_gufunc(
            _calculate_histogram_eager,
            f"({','.join(list(string.ascii_lowercase)[:n_axes])})->(nbins)",
            data,
            axes=[along_axes, (0,)],
            output_sizes={'nbins': len(bin_edges) - 1},
            along_axes=axes_in_chunk,
            bin_edges=bin_edges,
            bin_range=bin_range,
            normalization=normalization,
        )

    return hist


def _calculate_histogram_eager(
    data: np.ndarray,
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
    shape_rem_dims = tuple(data.shape[a] for a in remaining_dims)
    reshaped_data = reshaped_data.reshape(*shape_rem_dims, -1)

    # Apply vectorized version of np.histogram
    def _get_hist_values(arr):
        mask = np.ma.getmaskarray(arr)
        arr = arr[~mask]
        return np.histogram(arr, bins=bin_edges, range=bin_range)[0]

    v_histogram = np.vectorize(_get_hist_values, signature='(n)->(m)')
    hist = v_histogram(reshaped_data)

    # Mask points where all input data were masked (these are the ones where
    # the histograms sums to 0)
    hist_sum = hist.sum(axis=-1, keepdims=True)
    mask = np.isclose(hist_sum, 0.0)
    mask_broadcast = np.broadcast_to(mask, hist.shape)
    hist = np.ma.array(hist, mask=mask_broadcast)

    # Apply normalization
    if normalization == 'sum':
        hist = hist / np.ma.array(hist_sum, mask=mask)
    elif normalization == 'integral':
        diffs = np.ma.array(np.diff(bin_edges), dtype=data.dtype)
        hist = hist / np.ma.array(hist_sum, mask=mask) / diffs

    return hist


def _get_histogram_cube(
    cube: Cube,
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
        var_name=cube.var_name,
        long_name=cube.long_name,
        units=cube.units,
    )

    # Get result cube with correct dimensional metadata by using dummy
    # operation (max)
    hist_cube = cube.collapsed(coords, iris.analysis.MAX)
    hist_cube.cell_methods = [
        *cube.cell_methods, CellMethod('histogram', coords)
    ]
    hist_cube = add_leading_dim_to_cube(hist_cube, bin_coord)
    new_order = list(range(hist_cube.ndim))
    new_order[0] = hist_cube.ndim - 1
    new_order[-1] = 0
    hist_cube.transpose(new_order)

    # Adapt other metadata
    hist_cube.standard_name = None
    hist_cube.var_name = (
        'histogram' if hist_cube.var_name is None else
        f'histogram_{hist_cube.var_name}'
    )
    hist_cube.long_name = (
        'Histogram' if hist_cube.long_name is None else
        f'Histogram of {hist_cube.long_name}'
    )
    if normalization == 'integral':
        hist_cube.units = cube.units**-1
        hist_cube.attributes['normalization'] = 'integral'
    if normalization == 'sum':
        hist_cube.units = '1'
        hist_cube.attributes['normalization'] = 'sum'
    else:
        hist_cube.units = '1'
        hist_cube.attributes['normalization'] = 'none'

    return hist_cube
