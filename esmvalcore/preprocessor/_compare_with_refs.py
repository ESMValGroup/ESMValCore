"""Preprocessor functions for comparisons of data with reference datasets."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Literal, Optional

import dask
import dask.array as da
import iris.analysis
import iris.analysis.stats
import numpy as np
from iris.common.metadata import CubeMetadata
from iris.coords import CellMethod, Coord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.util import broadcast_to_shape
from scipy.stats import wasserstein_distance

from esmvalcore.preprocessor._area import _try_adding_calculated_cell_area
from esmvalcore.preprocessor._io import concatenate
from esmvalcore.preprocessor._other import get_array_module
from esmvalcore.preprocessor._time import get_time_weights

if TYPE_CHECKING:
    from esmvalcore.preprocessor import PreprocessorFile

logger = logging.getLogger(__name__)


BiasType = Literal['absolute', 'relative']


def bias(
    products: set[PreprocessorFile] | Iterable[Cube],
    ref_cube: Optional[Cube] = None,
    bias_type: BiasType = 'absolute',
    denominator_mask_threshold: float = 1e-3,
    keep_reference_dataset: bool = False,
) -> set[PreprocessorFile] | CubeList:
    """Calculate biases relative to a reference dataset.

    The reference dataset needs to be broadcastable to all input `products`.
    This supports `iris' rich broadcasting abilities
    <https://scitools-iris.readthedocs.io/en/stable/userguide/cube_maths.
    html#calculating-a-cube-anomaly>`__. To ensure this, the preprocessors
    :func:`esmvalcore.preprocessor.regrid` and/or
    :func:`esmvalcore.preprocessor.regrid_time` might be helpful.

    Notes
    -----
    The reference dataset can be specified with the `ref_cube` argument. If
    `ref_cube` is ``None``, exactly one input dataset in the `products` set
    needs to have the facet ``reference_for_bias: true`` defined in the recipe.
    Please do **not** specify the option `ref_cube` when using this
    preprocessor function in a recipe.

    Parameters
    ----------
    products:
        Input datasets/cubes for which the bias is calculated relative to a
        reference dataset/cube.
    ref_cube:
        Cube which is used as reference for the bias calculation. If ``None``,
        `products` needs to be a :obj:`set` of
        `~esmvalcore.preprocessor.PreprocessorFile` objects and exactly one
        dataset in `products` needs the facet ``reference_for_bias: true``. Do
        not specify this argument in a recipe.
    bias_type:
        Bias type that is calculated. Must be one of ``'absolute'`` (dataset -
        ref) or ``'relative'`` ((dataset - ref) / ref).
    denominator_mask_threshold:
        Threshold to mask values close to zero in the denominator (i.e., the
        reference dataset) during the calculation of relative biases. All
        values in the reference dataset with absolute value less than the given
        threshold are masked out. This setting is ignored when ``bias_type`` is
        set to ``'absolute'``. Please note that for some variables with very
        small absolute values (e.g., carbon cycle fluxes, which are usually
        :math:`< 10^{-6}` kg m :math:`^{-2}` s :math:`^{-1}`) it is absolutely
        essential to change the default value in order to get reasonable
        results.
    keep_reference_dataset:
        If ``True``, keep the reference dataset in the output. If ``False``,
        drop the reference dataset. Ignored if `ref_cube` is given.

    Returns
    -------
    set[PreprocessorFile] | CubeList
        Output datasets/cubes. Will be a :obj:`set` of
        :class:`~esmvalcore.preprocessor.PreprocessorFile` objects if
        `products` is also one, a :class:`~iris.cube.CubeList` otherwise.

    Raises
    ------
    ValueError
        Not exactly one input datasets contains the facet
        ``reference_for_bias: true`` if ``ref_cube=None``; ``ref_cube=None``
        and the input products are given as iterable of
        :class:`~iris.cube.Cube` objects; ``bias_type`` is not one of
        ``'absolute'`` or ``'relative'``.

    """
    ref_product = None
    all_cubes_given = all(isinstance(p, Cube) for p in products)

    # Get reference cube if not explicitly given
    if ref_cube is None:
        if all_cubes_given:
            raise ValueError(
                "A list of Cubes is given to this preprocessor; please "
                "specify a `ref_cube`"
            )
        (ref_cube, ref_product) = _get_ref(products, 'reference_for_bias')
    else:
        ref_product = None

    # Mask reference cube appropriately for relative biases
    if bias_type == 'relative':
        ref_cube = ref_cube.copy()
        npx = get_array_module(ref_cube.core_data())
        ref_cube.data = npx.ma.masked_inside(
            ref_cube.core_data(),
            -denominator_mask_threshold,
            denominator_mask_threshold,
        )

    # If input is an Iterable of Cube objects, calculate bias for each element
    if all_cubes_given:
        cubes = [_calculate_bias(c, ref_cube, bias_type) for c in products]
        return CubeList(cubes)

    # Otherwise, iterate over all input products, calculate bias and adapt
    # metadata and provenance information accordingly
    output_products = set()
    for product in products:
        if product == ref_product:
            continue
        cube = concatenate(product.cubes)

        # Calculate bias
        cube = _calculate_bias(cube, ref_cube, bias_type)

        # Adapt metadata and provenance information
        product.attributes['units'] = str(cube.units)
        if ref_product is not None:
            product.wasderivedfrom(ref_product)

        product.cubes = CubeList([cube])
        output_products.add(product)

    # Add reference dataset to output if desired
    if keep_reference_dataset and ref_product is not None:
        output_products.add(ref_product)

    return output_products


def _get_ref(products, ref_tag: str) -> tuple[Cube, PreprocessorFile]:
    """Get reference cube and product."""
    ref_products = []
    for product in products:
        if product.attributes.get(ref_tag, False):
            ref_products.append(product)
    if len(ref_products) != 1:
        raise ValueError(
            f"Expected exactly 1 dataset with '{ref_tag}: true', found "
            f"{len(ref_products):d}"
        )
    ref_product = ref_products[0]

    # Extract reference cube
    # Note: For technical reasons, product objects contain the member
    # ``cubes``, which is a list of cubes. However, this is expected to be a
    # list with exactly one element due to the call of concatenate earlier in
    # the preprocessing chain of ESMValTool. To make sure that this
    # preprocessor can also be used outside the ESMValTool preprocessing chain,
    # an additional concatenate call is added here.
    ref_cube = concatenate(ref_product.cubes)

    return (ref_cube, ref_product)


def _calculate_bias(cube: Cube, ref_cube: Cube, bias_type: BiasType) -> Cube:
    """Calculate bias for a single cube relative to a reference cube."""
    cube_metadata = cube.metadata

    if bias_type == 'absolute':
        cube = cube - ref_cube
        new_units = cube.units
    elif bias_type == 'relative':
        cube = (cube - ref_cube) / ref_cube
        new_units = '1'
    else:
        raise ValueError(
            f"Expected one of ['absolute', 'relative'] for bias_type, got "
            f"'{bias_type}'"
        )

    cube.metadata = cube_metadata
    cube.units = new_units

    return cube


MetricType = Literal[
    'weighted_rmse',
    'rmse',
    'weighted_pearsonr',
    'pearsonr',
    'emd',
]


def distance_metric(
    products: set[PreprocessorFile] | Iterable[Cube],
    metric: MetricType,
    ref_cube: Optional[Cube] = None,
    coords: Iterable[Coord] | Iterable[str] | None = None,
    keep_reference_dataset: bool = True,
    **kwargs,
) -> set[PreprocessorFile] | CubeList:
    """Calculate distance metrics.

    All input datasets need to have identical dimensional coordinates. This can
    for example be ensured with the preprocessors
    :func:`esmvalcore.preprocessor.regrid` and/or
    :func:`esmvalcore.preprocessor.regrid_time`.

    Notes
    -----
    This preprocessor requires a reference dataset, which can be specified with
    the `ref_cube` argument. If `ref_cube` is ``None``, exactly one input
    dataset in the `products` set needs to have the facet
    ``reference_for_metric: true`` defined in the recipe. Please do **not**
    specify the option `ref_cube` when using this preprocessor function in a
    recipe.

    Parameters
    ----------
    products:
        Input datasets/cubes for which the distance metric is calculated
        relative to a reference dataset/cube.
    metric:
        Distance metric that is calculated. Must be one of ``'weighted_rmse'``
        (weighted root mean square error), ``'rmse'`` (unweighted root mean
        square error), ``'weighted_pearsonr'`` (weighted Pearson correlation
        coefficient), ``'pearsonr'`` (unweighted Pearson correlation
        coefficient), ``'emd'`` (Earth mover's distance, also known as first
        Wasserstein metric W<sub>1</sub>).

        .. note::
            Metrics starting with `weighted_` will calculate weighted distance
            metrics if possible. Currently, the following `coords` (or any
            combinations that include them) will trigger weighting: `time`
            (will use lengths of time intervals as weights) and `latitude`
            (will use cell area weights). Time weights are always calculated
            from the input data. Area weights can be given as supplementary
            variables to the recipe (`areacella` or `areacello`, see
            :ref:`supplementary_variables`) or calculated from the input data
            (this only works for regular grids). By default, **NO**
            supplementary variables will be used; they need to be explicitly
            requested in the recipe.
    ref_cube:
        Cube which is used as reference for the distance metric calculation. If
        ``None``, `products` needs to be a :obj:`set` of
        :class:`~esmvalcore.preprocessor.PreprocessorFile` objects and exactly
        one dataset in `products` needs the facet ``reference_for_metric:
        true``. Do not specify this argument in a recipe.
    coords:
        Coordinates over which the distance metric is calculated. If ``None``,
        calculate the metric over all coordinates, which results in a scalar
        cube.
    keep_reference_dataset:
        If ``True``, also calculate the distance of the reference dataset with
        itself. If ``False``, drop the reference dataset.
    **kwargs:
        Additional options for the metric calculation. The following keyword
        arguments are supported:

        * `weighted_rmse` and `rmse`: none.
        * `weighted_pearsonr` and `pearsonr`: ``mdtol``, ``common_mask`` (all
          keyword arguments are passed to :func:`iris.analysis.stats.pearsonr`,
          see that link for more details on these arguments).
        * `emd`: ``nbins`` = number of bins used to create discrete probability
          mass function of data before calculating the EMD (:obj:`int`,
          default: 30).

    Returns
    -------
    set of esmvalcore.preprocessor.PreprocessorFile or iris.cube.CubeList
        Output datasets/cubes. Will be a :obj:`set` of
        :class:`~esmvalcore.preprocessor.PreprocessorFile` objects if
        `products` is also one, a :class:`~iris.cube.CubeList` otherwise.

    Raises
    ------
    ValueError
        Shape and coordinates of products and reference data does not match;
        not exactly one input datasets contains the facet
        ``reference_for_metric: true`` if ``ref_cube=None`; ``ref_cube=None``
        and the input products are given as iterable of
        :class:`~iris.cube.Cube` objects; an invalid ``metric`` has been given.
    iris.exceptions.CoordinateNotFoundError
        `longitude` is not found in cube if a weighted metric shall be
        calculated, `latitude` is in `coords`, and no `cell_area` is given
        as :ref:`supplementary_variables`.

    """
    reference_product = None
    all_cubes_given = all(isinstance(p, Cube) for p in products)

    # Get reference cube if not explicitly given
    if ref_cube is None:
        if all_cubes_given:
            raise ValueError(
                "A list of Cubes is given to this preprocessor; please "
                "specify a `ref_cube`"
            )
        reference_products = []
        for product in products:
            if product.attributes.get('reference_for_metric', False):
                reference_products.append(product)
        if len(reference_products) != 1:
            raise ValueError(
                f"Expected exactly 1 dataset with 'reference_for_metric: "
                f"true', found {len(reference_products):d}"
            )
        reference_product = reference_products[0]

        # Extract reference cube
        # Note: For technical reasons, product objects contain the member
        # ``cubes``, which is a list of cubes. However, this is expected to be
        # a list with exactly one element due to the call of concatenate
        # earlier in the preprocessing chain of ESMValTool. To make sure that
        # this preprocessor can also be used outside the ESMValTool
        # preprocessing chain, an additional concatenate call is added here.
        ref_cube = concatenate(reference_product.cubes)

    # If input is an Iterable of Cube objects, calculate distance metric for
    # each element
    if all_cubes_given:
        cubes = [
            _calculate_metric(c, ref_cube, metric, coords, **kwargs)
            for c in products
        ]
        return CubeList(cubes)

    # Otherwise, iterate over all input products, calculate metric and adapt
    # metadata and provenance information accordingly
    output_products = set()
    for product in products:
        if not keep_reference_dataset and product == reference_product:
            continue
        cube = concatenate(product.cubes)

        # Calculate distance metric
        cube = _calculate_metric(cube, ref_cube, metric, coords, **kwargs)

        # Adapt metadata and provenance information
        product.attributes['standard_name'] = cube.standard_name
        product.attributes['long_name'] = cube.long_name
        product.attributes['short_name'] = cube.var_name
        product.attributes['units'] = str(cube.units)
        if product != reference_product:
            product.wasderivedfrom(reference_product)

        product.cubes = CubeList([cube])
        output_products.add(product)

    return output_products


def _get_coords(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str] | None,
) -> Iterable[Coord] | Iterable[str]:
    """Get coordinates over which distance metric is calculated."""
    if coords is None:
        coords = [c.name() for c in cube.dim_coords]
        if len(coords) != cube.ndim:
            raise ValueError(
                f"If coords=None is specified, the cube "
                f"{cube.summary(shorten=True)} must not have unnamed "
                f"dimensions"
            )
    return coords


def _get_all_coord_dims(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
) -> tuple[int, ...]:
    all_coord_dims = []
    for coord in coords:
        all_coord_dims.extend(cube.coord_dims(coord))
    return tuple(set(all_coord_dims))


def _calculate_metric(
    cube: Cube,
    ref_cube: Cube,
    metric: MetricType,
    coords: Iterable[Coord] | Iterable[str] | None,
    **kwargs,
) -> Cube:
    """Calculate metric for a single cube relative to a reference cube."""
    # Make sure that dimensional metadata of data and ref data is compatible
    if cube.shape != ref_cube.shape:
        raise ValueError(
            f"Expected identical shapes of cube and reference cube for "
            f"distance metric calculation, got {cube.shape} and "
            f"{ref_cube.shape}, respectively"
        )
    try:
        cube + ref_cube  # dummy operation to check if cubes are compatible
    except Exception as exc:
        raise ValueError(
            "Cannot calculate distance metric between cube and reference cube "
        ) from exc

    # Perform the actual calculation of the distance metric
    # Note: we work on arrays here instead of cube to stay as flexible as
    # possible since some operations (e.g., sqrt()) are not available for cubes
    coords = _get_coords(cube, coords)
    metrics_funcs = {
        'weighted_rmse': partial(_calculate_rmse, weighted=True, **kwargs),
        'rmse': partial(_calculate_rmse, weighted=False, **kwargs),
        'weighted_pearsonr': partial(
            _calculate_pearsonr, weighted=True, **kwargs
        ),
        'pearsonr': partial(_calculate_pearsonr, weighted=False, **kwargs),
        'emd': partial(_calculate_emd, **kwargs),
    }
    if metric not in metrics_funcs:
        raise ValueError(
            f"Expected one of {list(metrics_funcs)} for metric, got '{metric}'"
        )
    (res_data, res_metadata) = metrics_funcs[metric](cube, ref_cube, coords)

    # Get result cube with correct dimensional metadata by using dummy
    # operation (max)
    res_cube = cube.collapsed(coords, iris.analysis.MAX)
    res_cube.data = res_data.astype(cube.dtype)
    res_cube.metadata = res_metadata
    res_cube.cell_methods = [*cube.cell_methods, CellMethod(metric, coords)]

    return res_cube


def _get_weights(
    cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
) -> da.Array:
    """Calculate weights for weighted distance metrics."""
    npx = get_array_module(cube.core_data())
    weights = npx.ones(cube.shape, dtype=cube.dtype)

    # Time weights: lengths of time interval
    if 'time' in coords:
        weights *= broadcast_to_shape(
            npx.array(get_time_weights(cube)),
            cube.shape,
            cube.coord_dims('time'),
        )

    # Latitude weights: cell areas
    if 'latitude' in coords:
        cube = cube.copy()  # avoid overwriting input cube
        if (
                not cube.cell_measures('cell_area') and
                not cube.coords('longitude')
        ):
            raise CoordinateNotFoundError(
                f"Cube {cube.summary(shorten=True)} needs a `longitude` "
                f"coordinate to calculate cell area weights for weighted "
                f"distance metric over coordinates {coords} (alternatively, "
                f"a `cell_area` can be given to the cube as supplementary "
                f"variable)"
            )
        _try_adding_calculated_cell_area(cube)
        weights *= broadcast_to_shape(
            cube.cell_measure('cell_area').core_data(),
            cube.shape,
            cube.cell_measure_dims('cell_area'),
        )

    return weights


def _calculate_rmse(
    cube: Cube,
    ref_cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
    *,
    weighted: bool,
) -> tuple[np.ndarray | da.Array, CubeMetadata]:
    """Calculate root mean square error."""
    # Data
    axis = _get_all_coord_dims(cube, coords)
    weights = _get_weights(cube, coords) if weighted else None
    squared_error = (cube.core_data() - ref_cube.core_data())**2
    npx = get_array_module(squared_error)
    rmse = npx.sqrt(npx.ma.average(squared_error, axis=axis, weights=weights))

    # Metadata
    metadata = CubeMetadata(
        None,
        'RMSE' if cube.long_name is None else f'RMSE of {cube.long_name}',
        'rmse' if cube.var_name is None else f'rmse_{cube.var_name}',
        cube.units,
        cube.attributes,
        cube.cell_methods,
    )

    return (rmse, metadata)


def _calculate_pearsonr(
    cube: Cube,
    ref_cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
    *,
    weighted: bool,
    **kwargs,
) -> tuple[np.ndarray | da.Array, CubeMetadata]:
    """Calculate Pearson correlation coefficient."""
    # Data
    weights = _get_weights(cube, coords) if weighted else None
    res_cube = iris.analysis.stats.pearsonr(
        cube, ref_cube, corr_coords=coords, weights=weights, **kwargs
    )

    # Metadata
    metadata = CubeMetadata(
        None,
        (
            "Pearson's r" if cube.long_name is None
            else f"Pearson's r of {cube.long_name}"
        ),
        'pearsonr' if cube.var_name is None else f'pearsonr_{cube.var_name}',
        '1',
        cube.attributes,
        cube.cell_methods,
    )

    return (res_cube.core_data(), metadata)


def _calculate_emd(
    cube: Cube,
    ref_cube: Cube,
    coords: Iterable[Coord] | Iterable[str],
    *,
    n_bins: int = 30,
) -> tuple[np.ndarray | da.Array, CubeMetadata]:
    """Calculate Earth mover's distance."""
    # Data
    axis = _get_all_coord_dims(cube, coords)
    (bins, bin_centers) = _get_bins(cube, ref_cube, n_bins)

    emd = _calculate_emd_eager(
        cube.data,
        ref_cube.data,
        axis=axis,
        bins=bins,
        bin_centers=bin_centers,
    )

    # Metadata
    metadata = CubeMetadata(
        None,
        'EMD' if cube.long_name is None else f'EMD of {cube.long_name}',
        'emd' if cube.var_name is None else f'emd_{cube.var_name}',
        '1',
        cube.attributes,
        cube.cell_methods,
    )

    return (emd, metadata)


def _get_bins(
    cube: Cube,
    ref_cube: Cube,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get bins for discretization of data."""
    all_data = da.stack([cube.core_data(), ref_cube.core_data()])
    (min_, max_) = dask.compute(all_data.min(), all_data.max())
    small_value = (max_ - min_) * 0.01 / n_bins
    bins = np.linspace(min_ - small_value, max_ + small_value, n_bins + 1)
    bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2.0
    return (bins, bin_centers)


def _calculate_emd_eager(
    data: np.ndarray,
    ref_data: np.ndarray,
    *,
    axis: Optional[int | tuple[int, ...]],
    bins: np.ndarray,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """Calculate Earth mover's distance along axis (eager version)."""
    pmf = _get_pmf(data, axis, bins)
    ref_pmf = _get_pmf(ref_data, axis, bins)

    # Get vectorized version of scipy.stats.wasserstein_distance and calculate
    # EMD metric (= Wasserstein distance)
    v_wasserstein_distance = np.vectorize(
        wasserstein_distance, signature='(n),(n),(n),(n)->()'
    )
    emd = v_wasserstein_distance(bin_centers, bin_centers, pmf, ref_pmf)

    return emd


def _get_pmf(
    data: np.ndarray,
    axis: Optional[int | tuple[int, ...]],
    bins: np.ndarray,
) -> np.ndarray:
    """Get probaility mass function (PMF) of data along axis.

    This will return an array of shape `(x1, x2, ..., n_bins)` where `xi` are
    the dimensions of `data` not appearing in `axis` and `n_bins` is the number
    of bins.

    """
    if axis is None:
        axis = tuple(range(data.ndim))
    elif isinstance(axis, int):
        axis = tuple([axis])

    # Create array with shape (x1, x2, ..., y) where the `xi` are the
    # dimensions of `data` not in `axis` and `y` is the product of the
    # remaining dimensions
    remaining_dims = tuple(a for a in range(data.ndim) if a not in axis)
    shape_rem_dims = tuple(data.shape[a] for a in remaining_dims)
    reshaped_data = data.reshape(*shape_rem_dims, -1)

    # Use vectorized version of np.histogram to get PMF (which has been
    # normalized by number of samples that entered the histogram calculation,
    # i.e., `y` from above)
    def _get_hist_values(*args, **kwargs):
        return np.histogram(*args, **kwargs)[0]

    v_histogram = np.vectorize(
        _get_hist_values, excluded=('bins', 'range'), signature='(n)->(m)'
    )
    pmf = v_histogram(
        reshaped_data, bins=bins, range=(bins[0], bins[-1])
    ) / reshaped_data.shape[-1]

    return pmf
