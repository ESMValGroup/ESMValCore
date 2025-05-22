from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import dask.array as da
import ibicus
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._io import concatenate
from esmvalcore.preprocessor._shared import (
    preserve_float_dtype,
)
from esmvalcore.preprocessor._time import extract_time

if TYPE_CHECKING:
    from esmvalcore.preprocessor import PreprocessorFile

logger = logging.getLogger(__name__)

IBICUS_DOCS = "https://ibicus.readthedocs.io/en/latest/reference/"


def _create_new_product(product, filename):
    """Create a new product .

    Returns a new PreprocessorFile with same attributes and settings as the
    input product but for a different filename.
    NOTE: There might be a better way to do this. If not, we should such a
    method to the PreprocessorFile class.
    """
    new_product = product.__class__(
        filename,
        product.attributes,
        product.settings,
        product.datasets,
    )
    new_product.cubes = product.cubes
    new_product.initialize_provenance(product.activity)
    return new_product


def bias_adjust(
    products: set[PreprocessorFile] | Iterable[Cube],
    method: str,
    reference: Optional[Cube] = None,
    # reference_period: Optional[dict] = None,
    # projection_period: Optional[dict] = None,
    # debias_kwargs: Optional[dict] = None,
    # apply_kwargs: Optional[dict] = None,
    keep_original_datasets: bool = False,
    **kwargs,
) -> Cube:
    """Apply bias adjustment to a cube using the specified method.

    Notes
    -----
    This preprocessor requires a reference dataset, which can be specified with
    the `reference` argument. If `reference` is ``None``, exactly one input
    dataset in the `products` set needs to have the facet
    ``reference_for_metric: true`` defined in the recipe. Please do **not**
    specify the option `reference` when using this preprocessor function in a
    recipe.

    Parameters
    ----------
    products:
        Input datasets/cubes to which the bias adjustment will be applied.
    method:
        The bias adjustment method to use. Debias classes available in
        the ``ibicus.debias`` module are valid methods:

        * ``"LinearScaling"``
        * ``"DeltaChange"``
        * ``"QuantileMapping"``
        * ``"ScaledDistributionMapping"``
        * ``"CDFt"``
        * ``"ECDFM"``
        * ``"QuantileDeltaMapping"``
        * ``"ISIMIP"``

        See https://ibicus.readthedocs.io/en/latest/reference/debias.html for
        more information.
    reference:
        Reference dataset/cube to use for bias adjustment. If ``None``,
        `products` needs to be a :obj:`set` of
        :class:`~esmvalcore.preprocessor.PreprocessorFile` objects and exactly
        one dataset in `products` needs the facet
        ``reference_for_bias_adjustment:true``.
        Do not specify this argument in a recipe.
    keep_original_datasets:
        If ``True``, the original input and reference datasets will be added to
        the output.
        Default is ``False``.
    **kwargs:
        Additional keyword arguments to customize the bias adjustment.
        Possible parameters are:
        reference_period:
            Dictionary with the reference period to use for bias adjustment.
            The given time period will be extracted from the reference and target
            datasets. The dict must contain ``start_year`` and ``end_year`` keys.
            ``start_month=1``, ``start_day=1``, ``end_month=12`` and ``end_day=31``
            are optional.
            If ``None``, the full period of the reference dataset will be used.
        debias_kwargs:
            Dictionary with additional keyword arguments for the debiasing
            method. The keys depend on the method used.
            See https://ibicus.readthedocs.io/en/latest/reference/debias.html
            for more information.
        apply_kwargs:
            Dictionary with additional keyword arguments passed to the
            ``apply()``-method of the debias class.
        variable:
            The debiaser is initialized with default parameters for the cubes
            variable. If this parameter is given, it is used instead of
            cube.var_name.

    Returns
    -------
    set of esmvalcore.preprocessor.PreprocessorFile or iris.cube.CubeList
        Output datasets/cubes. Will be a :obj:`set` of
        :class:`~esmvalcore.preprocessor.PreprocessorFile` objects if
        `products` is also one, a :class:`~iris.cube.CubeList` otherwise.
    """
    reference_product = None
    all_cubes_given = all(isinstance(p, Cube) for p in products)

    # Get reference cube if not explicitly given
    if reference is None:
        if all_cubes_given:
            raise ValueError(
                "A list of Cubes is given to this preprocessor; please "
                "specify a `reference`"
            )
        reference, reference_product = _get_ref(
            products, "reference_for_bias_adjustment"
        )

    # If input is an Iterable of Cube objects, calculate distance metric for
    # each element (usually not via recipe)
    if all_cubes_given:
        cubes = [
            _apply_debiaser(c, reference, method, **kwargs) for c in products
        ]
        logger.warning("debiaser applied")
        return CubeList(cubes)

    # Otherwise, iterate over all input products, calculate metric and adapt
    # metadata and provenance information accordingly (usually from recipe)
    output_products = set()
    for product in products:
        if keep_original_datasets:
            output_products.add(product)
        if product == reference_product:
            continue
        # create a new product for the debiased data
        fname = Path(product.filename)
        filename = fname.with_name(fname.stem + "_adjusted" + fname.suffix)
        # TODO: regenerate filename from settings or cube data? correct dates
        new_product = _create_new_product(product, filename)
        new_product.wasderivedfrom(reference_product)
        new_product.wasderivedfrom(product)
        if new_product.attributes is None:
            logger.error("No attributes found in product. Skipping.")
            continue
        cube = concatenate(new_product.cubes)
        # Adapt metadata and provenance information
        # TODO: attributes are automatically updated from cube on save (before saving prov)
        new_product.attributes["standard_name"] = cube.standard_name
        new_product.attributes["long_name"] = cube.long_name
        new_product.attributes["short_name"] = cube.var_name
        new_product.attributes["units"] = str(cube.units)
        new_product.attributes["is_bias_adjusted"] = True

        cube = _apply_debiaser(cube, reference, method, **kwargs)
        new_product.cubes = CubeList([cube])
        output_products.add(new_product)

    return output_products


@preserve_float_dtype
def _apply_debiaser(
    cube: Cube, reference: Cube, method: str, **kwargs
) -> Cube:
    """Apply the debiaser to the projection."""
    debias_kwargs = kwargs.get("debias_kwargs", {})
    apply_kwargs = kwargs.get("apply_kwargs", {})
    debiaser = _setup_debiaser(cube, method, **debias_kwargs)
    # setup chunks for parallel processing with dask
    # cube = rechunk_cube(cube, ["time"], "auto")  # did not work as expected
    # chunking = (-1, 20, 20)  # test with actual chunking
    chunking = (-1, "auto", "auto")  # debiasers do now allow chunking time dim
    cube.data = cube.core_data().rechunk(chunking)
    reference_period = kwargs.get("reference_period", None)
    projection_period = kwargs.get("projection_period", None)
    ref, hist, fut = _select_time_ranges(
        cube,
        reference,
        reference_period=reference_period,
        projection_period=projection_period,
    )
    ref.data = ref.core_data().rechunk(chunking)  # must match chunks of hist
    debiased = fut.copy()
    # debiased_data = debiaser.apply(ref, hist, fut)
    # debiased.data = debiased_data
    debiased.data = da.map_blocks(
        debiaser.apply,
        ref.core_data(),
        hist.core_data(),
        fut.core_data(),
        progressbar=False,
        parallel=False,
        dtype=debiased.dtype,
        chunks=fut.core_data().chunks,
        **apply_kwargs,  # does it wok like this in map_blocks?
    )
    logger.info("Result is lazy: %s", debiased.has_lazy_data())
    return debiased


def _time_range_from_cube(cube: Cube) -> dict:
    """Get time range from cube."""
    start_time = cube.coord("time").cell(0).point
    end_time = cube.coord("time").cell(-1).point
    return {
        "start_year": start_time.year,
        "start_month": start_time.month,
        "start_day": start_time.day,
        "end_year": end_time.year,
        "end_month": end_time.month,
        "end_day": end_time.day,
    }


def _fill_and_validate_time_range(period: dict) -> None:
    if not all(k in period for k in ("start_year", "end_year")):
        raise ValueError(
            "Period must contain at least start_year and end_years"
        )
    period.setdefault("start_month", 1)
    period.setdefault("start_day", 1)
    period.setdefault("end_month", 12)
    period.setdefault("end_day", 31)


def _select_time_ranges(
    cube: Cube,
    reference: Cube,
    reference_period: Optional[dict] = None,
    projection_period: Optional[dict] = None,
) -> Cube:
    """Select time ranges from a cube."""
    # ensure complete period dicts for extraction
    if reference_period is None:
        reference_period = _time_range_from_cube(reference)
    _fill_and_validate_time_range(reference_period)
    if projection_period is None:
        projection_period = _time_range_from_cube(cube)
    _fill_and_validate_time_range(projection_period)
    # select cubes for given time ranges
    reference = extract_time(reference, **reference_period)
    historical = extract_time(cube, **reference_period)
    future = extract_time(cube, **projection_period)
    return reference, historical, future


def _setup_debiaser(
    cube: Cube, method: str, **kwargs
) -> ibicus.debias.Debiaser:
    # TODO: apply kwargs, variables etc..
    debias_module = importlib.import_module("ibicus.debias")
    try:
        debiaser = getattr(debias_module, method)
    except AttributeError as exc:
        msg = (
            f"The requested bias adjustment method '{method}' is not available. "
            f"Please check the documentation at {IBICUS_DOCS}/debias.html for "
            "available methods and provide a valid class name."
        )
        logger.error(msg)
        raise AttributeError(msg) from exc
    variable = kwargs.get("variable", cube.var_name)
    logger.info("Initialising debiaser for variable %s", variable)
    return debiaser.from_variable(variable)


def _get_ref(products, ref_tag: str) -> tuple[Cube, PreprocessorFile]:
    """Get reference cube and product."""
    ref_products = []
    for product in products:
        logger.warning(product.attributes)
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
    reference = concatenate(ref_product.cubes)

    return (reference, ref_product)
