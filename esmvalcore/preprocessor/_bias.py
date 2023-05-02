"""Preprocessor functions to calculate biases from data."""
import logging

import dask.array as da
import iris.cube

from ._io import concatenate

logger = logging.getLogger(__name__)


def bias(products, bias_type='absolute', denominator_mask_threshold=1e-3,
         keep_reference_dataset=False):
    """Calculate biases.

    Notes
    -----
    This preprocessor requires a reference dataset. For this, exactly one input
    dataset needs to have the facet ``reference_for_bias: true`` defined in the
    recipe. In addition, all input datasets need to have identical dimensional
    coordinates. This can for example be ensured with the preprocessors
    :func:`esmvalcore.preprocessor.regrid` and/or
    :func:`esmvalcore.preprocessor.regrid_time`.

    Parameters
    ----------
    products: set of esmvalcore.preprocessor.PreprocessorFile
        Input datasets. Exactly one datasets needs the facet
        ``reference_for_bias: true``.
    bias_type: str, optional (default: 'absolute')
        Bias type that is calculated. Must be one of ``'absolute'`` (dataset -
        ref) or ``'relative'`` ((dataset - ref) / ref).
    denominator_mask_threshold: float, optional (default: 1e-3)
        Threshold to mask values close to zero in the denominator (i.e., the
        reference dataset) during the calculation of relative biases. All
        values in the reference dataset with absolute value less than the given
        threshold are masked out. This setting is ignored when ``bias_type`` is
        set to ``'absolute'``. Please note that for some variables with very
        small absolute values (e.g., carbon cycle fluxes, which are usually
        :math:`< 10^{-6}` kg m :math:`^{-2}` s :math:`^{-1}`) it is absolutely
        essential to change the default value in order to get reasonable
        results.
    keep_reference_dataset: bool, optional (default: False)
        If ``True``, keep the reference dataset in the output. If ``False``,
        drop the reference dataset.

    Returns
    -------
    set of esmvalcore.preprocessor.PreprocessorFile
        Output datasets.

    Raises
    ------
    ValueError
        Not exactly one input datasets contains the facet
        ``reference_for_bias: true``; ``bias_type`` is not one of
        ``'absolute'`` or ``'relative'``.

    """
    # Get reference product
    reference_product = []
    for product in products:
        if product.attributes.get('reference_for_bias', False):
            reference_product.append(product)
    if len(reference_product) != 1:
        raise ValueError(
            f"Expected exactly 1 dataset with 'reference_for_bias: true', "
            f"found {len(reference_product):d}")
    reference_product = reference_product[0]

    # Extract reference cube
    # Note: For technical reasons, product objects contain the member
    # ``cubes``, which is a list of cubes. However, this is expected to be a
    # list with exactly one element due to the call of concatenate earlier in
    # the preprocessing chain of ESMValTool. To make sure that this
    # preprocessor can also be used outside the ESMValTool preprocessing chain,
    # an additional concatenate call is added here.
    ref_cube = concatenate(reference_product.cubes)
    if bias_type == 'relative':
        ref_cube = ref_cube.copy()
        ref_cube.data = da.ma.masked_inside(ref_cube.core_data(),
                                            -denominator_mask_threshold,
                                            denominator_mask_threshold)

    # Iterate over all input datasets and calculate bias
    output_products = set()
    for product in products:
        if product == reference_product:
            continue
        cube = concatenate(product.cubes)
        cube_metadata = cube.metadata

        # Calculate bias
        if bias_type == 'absolute':
            cube = cube - ref_cube
            new_units = str(cube.units)
        elif bias_type == 'relative':
            cube = (cube - ref_cube) / ref_cube
            new_units = '1'
        else:
            raise ValueError(
                f"Expected one of ['absolute', 'relative'] for bias_type, got "
                f"'{bias_type}'")

        # Adapt cube metadata and provenance information
        cube.metadata = cube_metadata
        cube.units = new_units
        product.attributes['units'] = new_units
        product.wasderivedfrom(reference_product)

        product.cubes = iris.cube.CubeList([cube])
        output_products.add(product)

    # Add reference dataset to output if desired
    if keep_reference_dataset:
        output_products.add(reference_product)

    return output_products
