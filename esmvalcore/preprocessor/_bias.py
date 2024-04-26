"""Preprocessor functions to calculate biases from data."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Optional

import dask.array as da
from iris.cube import Cube, CubeList

from ._io import concatenate

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
        dataset in `products` needs the facet ``reference_for_bias: true``.
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
        ref_cube.data = da.ma.masked_inside(
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
