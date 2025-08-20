"""Preprocessor module."""

from __future__ import annotations

import copy
import inspect
import logging
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, TypeAlias

from iris.cube import Cube, CubeList

from esmvalcore._provenance import TrackedFile
from esmvalcore._task import BaseTask
from esmvalcore.cmor.check import cmor_check_data, cmor_check_metadata
from esmvalcore.cmor.fix import fix_data, fix_file, fix_metadata
from esmvalcore.preprocessor._area import (
    area_statistics,
    extract_named_regions,
    extract_region,
    extract_shape,
    meridional_statistics,
    zonal_statistics,
)
from esmvalcore.preprocessor._compare_with_refs import bias, distance_metric
from esmvalcore.preprocessor._concatenate import concatenate
from esmvalcore.preprocessor._cycles import amplitude
from esmvalcore.preprocessor._dask_progress import _compute_with_progress
from esmvalcore.preprocessor._derive import derive
from esmvalcore.preprocessor._detrend import detrend
from esmvalcore.preprocessor._io import (
    _get_debug_filename,
    _sort_products,
    load,
    save,
    write_metadata,
)
from esmvalcore.preprocessor._mask import (
    mask_above_threshold,
    mask_below_threshold,
    mask_fillvalues,
    mask_glaciated,
    mask_inside_range,
    mask_landsea,
    mask_landseaice,
    mask_multimodel,
    mask_outside_range,
)
from esmvalcore.preprocessor._multimodel import (
    ensemble_statistics,
    multi_model_statistics,
)
from esmvalcore.preprocessor._other import (
    align_metadata,
    clip,
    cumulative_sum,
    histogram,
)
from esmvalcore.preprocessor._regrid import (
    extract_coordinate_points,
    extract_levels,
    extract_location,
    extract_point,
    regrid,
)
from esmvalcore.preprocessor._rolling_window import rolling_window_statistics
from esmvalcore.preprocessor._supplementary_vars import (
    add_supplementary_variables,
    remove_supplementary_variables,
)
from esmvalcore.preprocessor._time import (
    annual_statistics,
    anomalies,
    climate_statistics,
    clip_timerange,
    daily_statistics,
    decadal_statistics,
    extract_month,
    extract_season,
    extract_time,
    hourly_statistics,
    local_solar_time,
    monthly_statistics,
    regrid_time,
    resample_hours,
    resample_time,
    seasonal_statistics,
    timeseries_filter,
)
from esmvalcore.preprocessor._trend import linear_trend, linear_trend_stderr
from esmvalcore.preprocessor._units import accumulate_coordinate, convert_units
from esmvalcore.preprocessor._volume import (
    axis_statistics,
    depth_integration,
    extract_surface_from_atm,
    extract_trajectory,
    extract_transect,
    extract_volume,
    volume_statistics,
)
from esmvalcore.preprocessor._weighting import weighting_landsea_fraction

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from dask.delayed import Delayed

    from esmvalcore.dataset import Dataset, File

logger = logging.getLogger(__name__)

__all__ = [
    # File reformatting/CMORization
    "fix_file",
    # Load cubes from file
    "load",
    # Metadata reformatting/CMORization
    "fix_metadata",
    # Concatenate all cubes in one
    "concatenate",
    "cmor_check_metadata",
    # Extract years given by dataset keys (timerange/start_year and end_year)
    "clip_timerange",
    # Data reformatting/CMORization
    "fix_data",
    "cmor_check_data",
    # Attach ancillary variables and cell measures
    "add_supplementary_variables",
    # Derive variable
    "derive",
    # Align metadata
    "align_metadata",
    # Time extraction (as defined in the preprocessor section)
    "extract_time",
    "extract_season",
    "extract_month",
    "resample_hours",
    "resample_time",
    # Level extraction
    "extract_levels",
    # Extract surface
    "extract_surface_from_atm",
    # Weighting
    "weighting_landsea_fraction",
    # Mask landsea (fx or Natural Earth)
    "mask_landsea",
    # Natural Earth only
    "mask_glaciated",
    # Mask landseaice, sftgif only
    "mask_landseaice",
    # Regridding
    "regrid",
    # Point interpolation
    "extract_coordinate_points",
    "extract_point",
    "extract_location",
    # Masking missing values
    "mask_multimodel",
    "mask_fillvalues",
    "mask_above_threshold",
    "mask_below_threshold",
    "mask_inside_range",
    "mask_outside_range",
    # Other
    "clip",
    "rolling_window_statistics",
    "cumulative_sum",
    # Region operations
    "extract_region",
    "extract_shape",
    "extract_volume",
    "extract_trajectory",
    "extract_transect",
    "detrend",
    "extract_named_regions",
    "axis_statistics",
    "depth_integration",
    "area_statistics",
    "volume_statistics",
    # Time operations
    "local_solar_time",
    "amplitude",
    "zonal_statistics",
    "meridional_statistics",
    "accumulate_coordinate",
    "hourly_statistics",
    "daily_statistics",
    "monthly_statistics",
    "seasonal_statistics",
    "annual_statistics",
    "decadal_statistics",
    "climate_statistics",
    "anomalies",
    "regrid_time",
    "timeseries_filter",
    "linear_trend",
    "linear_trend_stderr",
    # Convert units
    "convert_units",
    # Histograms
    "histogram",
    # Ensemble statistics
    "ensemble_statistics",
    # Multi model statistics
    "multi_model_statistics",
    # Comparison with reference datasets
    "bias",
    "distance_metric",
    # Remove supplementary variables from cube
    "remove_supplementary_variables",
    # Save to file
    "save",
]

TIME_PREPROCESSORS: list[str] = [
    "clip_timerange",
    "extract_time",
    "extract_season",
    "extract_month",
    "daily_statistics",
    "monthly_statistics",
    "seasonal_statistics",
    "annual_statistics",
    "decadal_statistics",
    "climate_statistics",
    "anomalies",
    "regrid_time",
]

DEFAULT_ORDER: tuple[str, ...] = tuple(__all__)
"""
By default, preprocessor functions are applied in this order.
"""

# The order of initial and final steps cannot be configured
INITIAL_STEPS: tuple[str, ...] = DEFAULT_ORDER[
    : DEFAULT_ORDER.index("add_supplementary_variables") + 1
]
FINAL_STEPS: tuple[str, ...] = DEFAULT_ORDER[
    DEFAULT_ORDER.index("remove_supplementary_variables") :
]

MULTI_MODEL_FUNCTIONS: set[str] = {
    "bias",
    "distance_metric",
    "ensemble_statistics",
    "multi_model_statistics",
    "mask_multimodel",
    "mask_fillvalues",
}


def _get_itype(step: str) -> str:
    """Get the input type of a preprocessor function."""
    function = globals()[step]
    return next(iter(inspect.signature(function).parameters))


def check_preprocessor_settings(settings: dict[str, Any]) -> None:
    """Check preprocessor settings."""
    for step, kwargs in settings.items():
        if step not in DEFAULT_ORDER:
            msg = (
                f"Unknown preprocessor function '{step}', choose from: "
                f"{', '.join(DEFAULT_ORDER)}"
            )
            raise ValueError(msg)

        function = globals()[step]

        # Note: below, we do not use inspect.getfullargspec since this does not
        # work with decorated functions. On the other hand, inspect.signature
        # behaves correctly with properly decorated functions (those that use
        # functools.wraps).
        signature = inspect.signature(function)
        args = [
            n
            for (n, p) in signature.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ][1:]

        # Check for invalid arguments (only possible if no *args or **kwargs
        # allowed)
        var_kinds = [p.kind for p in signature.parameters.values()]
        check_args = not any(
            [
                inspect.Parameter.VAR_POSITIONAL in var_kinds,
                inspect.Parameter.VAR_KEYWORD in var_kinds,
            ],
        )
        if check_args:
            invalid_args = set(kwargs) - set(args)
            if invalid_args:
                msg = (
                    f"Invalid argument(s) [{', '.join(invalid_args)}] "
                    f"encountered for preprocessor function {step}. \n"
                    f"Valid arguments are: [{', '.join(args)}]"
                )
                raise ValueError(msg)

        # Check for missing arguments
        defaults = [
            p.default
            for p in signature.parameters.values()
            if p.default is not inspect.Parameter.empty
        ]
        end = None if not defaults else -len(defaults)
        missing_args = set(args[:end]) - set(kwargs)
        if missing_args:
            msg = (
                f"Missing required argument(s) {missing_args} for "
                f"preprocessor function {step}"
            )
            raise ValueError(msg)

        # Final sanity check in case the above fails to catch a mistake
        try:
            signature.bind(None, **kwargs)
        except TypeError:
            logger.error(
                "Wrong preprocessor function arguments in function '%s'",
                step,
            )
            raise


def _check_multi_model_settings(products: Iterable[PreprocessorFile]) -> None:
    """Check that multi dataset settings are identical for all products."""
    multi_model_steps = (
        step
        for step in MULTI_MODEL_FUNCTIONS
        if any(step in p.settings for p in products)
    )
    for step in multi_model_steps:
        reference = None
        for product in products:
            settings = product.settings.get(step)
            if settings is None:
                continue
            if reference is None:
                reference = product
            elif reference.settings[step] != settings:
                msg = (
                    "Unable to combine differing multi-dataset settings for "
                    f"{reference.filename} and {product.filename}, "
                    f"{reference.settings[step]} and {settings}"
                )
                raise ValueError(msg)


def _get_multi_model_settings(
    products: set[PreprocessorFile],
    step: str,
) -> tuple[dict[str, Any], set[PreprocessorFile]]:
    """Select settings for multi model step."""
    _check_multi_model_settings(products)
    settings = {}
    exclude: set[PreprocessorFile] = set()
    for product in products:
        if step in product.settings:
            settings = product.settings[step]
        else:
            exclude.add(product)
    return settings, exclude


def _run_preproc_function(
    function: Callable,
    items: PreprocessorItem | Sequence[PreprocessorItem],
    kwargs: Any,
    input_files: Sequence[File] | None = None,
) -> PreprocessorItem | Sequence[PreprocessorItem]:
    """Run preprocessor function."""
    kwargs_str = ",\n".join(
        [f"{k} = {pformat(v)}" for (k, v) in kwargs.items()],
    )
    if input_files is None:
        file_msg = ""
    else:
        file_msg = (
            f"\nloaded from original input file(s)\n{pformat(input_files)}"
        )
    logger.debug(
        "Running preprocessor function '%s' on the data\n%s%s\nwith function "
        "argument(s)\n%s",
        function.__name__,
        pformat(items),
        file_msg,
        kwargs_str,
    )
    try:
        return function(items, **kwargs)
    except Exception:
        # To avoid very long error messages, we truncate the arguments and
        # input files here at a given threshold
        n_shown_args = 4
        if input_files is not None and len(input_files) > n_shown_args:
            n_not_shown_files = len(input_files) - n_shown_args
            file_msg = (
                f"\nloaded from original input file(s)\n"
                f"{pformat(input_files[:n_shown_args])}\n(and "
                f"{n_not_shown_files:d} further file(s) not shown "
                f"here; refer to the debug log for a full list)"
            )

        # Make sure that the arguments are indexable
        if isinstance(items, (PreprocessorFile, Cube, str, Path)):
            items = [items]
        if isinstance(items, set):
            items = list(items)

        if len(items) <= n_shown_args:
            data_msg = pformat(items)
        else:
            n_not_shown_args = len(items) - n_shown_args
            data_msg = (
                f"{pformat(items[:n_shown_args])}\n(and "
                f"{n_not_shown_args:d} further argument(s) not shown "
                f"here; refer to the debug log for a full list)"
            )
        logger.error(
            "Failed to run preprocessor function '%s' on the data\n%s%s\nwith "
            "function argument(s)\n%s",
            function.__name__,
            data_msg,
            file_msg,
            kwargs_str,
        )
        raise


def preprocess(
    items: Sequence[PreprocessorItem],
    step: str,
    input_files: list[File] | None = None,
    output_file: Path | None = None,
    debug: bool = False,
    **settings: Any,
) -> list[PreprocessorItem]:
    """Run preprocessor."""
    logger.debug("Running preprocessor step %s", step)
    function = globals()[step]
    itype = _get_itype(step)

    for item in items:
        if isinstance(item, Cube) and item.has_lazy_data():
            item.data = item.core_data().rechunk()

    result = []
    if itype.endswith("s"):
        result.append(
            _run_preproc_function(
                function,
                items,
                settings,
                input_files=input_files,
            ),
        )
    else:
        for item in items:
            result.append(
                _run_preproc_function(
                    function,
                    item,
                    settings,
                    input_files=input_files,
                ),
            )

    if step == "save":
        return result

    items = []
    for item in result:
        if isinstance(item, (PreprocessorFile, Cube, str, Path)):
            items.append(item)
        else:
            items.extend(item)

    if debug:
        logger.debug("Result %s", items)
        if all(isinstance(elem, Cube) for elem in items):
            filename = _get_debug_filename(output_file, step)
            save(items, filename)

    return items


def get_step_blocks(
    steps: Iterable[str],
    order: Sequence[str],
) -> list[list[str]]:
    """Group steps into execution blocks."""
    blocks: list[list[str]] = []
    prev_step_type = None
    for step in order[len(INITIAL_STEPS) : -len(FINAL_STEPS)]:
        if step in steps:
            step_type = step in MULTI_MODEL_FUNCTIONS
            if step_type is not prev_step_type:
                block: list[str] = []
                blocks.append(block)
            prev_step_type = step_type
            block.append(step)
    return blocks


class PreprocessorFile(TrackedFile):
    """Preprocessor output file."""

    def __init__(
        self,
        filename: Path,
        attributes: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        datasets: list | None = None,
    ) -> None:
        if datasets is not None:
            # Load data using a Dataset
            input_files = []
            for dataset in datasets:
                input_files.extend(dataset.files)
                for supplementary in dataset.supplementaries:
                    input_files.extend(supplementary.files)
            ancestors = [TrackedFile(f) for f in input_files]
        else:
            # Multimodel preprocessor functions set ancestors at runtime
            # instead of here.
            input_files = []
            ancestors = []

        self.datasets: list[Dataset] | None = datasets
        self._cubes: CubeList | None = None
        self._input_files = input_files

        # Set some preprocessor settings (move all defaults here?)
        if settings is None:
            settings = {}
        self.settings = copy.deepcopy(settings)
        if attributes is None:
            attributes = {}
        attributes = copy.deepcopy(attributes)
        if "save" not in self.settings:
            self.settings["save"] = {}
        self.settings["save"]["filename"] = filename

        attributes["filename"] = filename

        super().__init__(
            filename=filename,
            attributes=attributes,
            ancestors=ancestors,
        )

    def check(self) -> None:
        """Check preprocessor settings."""
        check_preprocessor_settings(self.settings)

    def apply(self, step: str, debug: bool = False) -> None:
        """Apply preprocessor step to product."""
        if step not in self.settings:
            msg = f"{self} has no settings for step {step}"
            raise ValueError(msg)
        self.cubes = preprocess(
            self.cubes,
            step,
            input_files=self._input_files,
            output_file=self.filename,
            debug=debug,
            **self.settings[step],
        )

    @property
    def cubes(self) -> list[Cube]:
        """Cubes."""
        if self._cubes is None:
            self._cubes = [ds.load() for ds in self.datasets]  # type: ignore
        return self._cubes

    @cubes.setter
    def cubes(self, value: CubeList) -> None:
        self._cubes = value

    def save(self) -> Delayed | None:
        """Save cubes to disk."""
        return preprocess(
            self._cubes,  # type: ignore
            "save",
            input_files=self._input_files,
            **self.settings["save"],
        )[0]

    def close(self) -> Delayed | None:
        """Close the file."""
        result = None
        if self._cubes is not None:
            self._update_attributes()
            result = self.save()
            self._cubes = None
            self.save_provenance()
        return result

    def _update_attributes(self) -> None:
        """Update product attributes from cube metadata."""
        if not self._cubes:
            return
        ref_cube = self._cubes[0]

        # Names
        names = {
            "standard_name": "standard_name",
            "long_name": "long_name",
            "var_name": "short_name",
        }
        for name_in, name_out in names.items():
            cube_val = getattr(ref_cube, name_in)
            self.attributes[name_out] = "" if cube_val is None else cube_val

        # Units
        self.attributes["units"] = str(ref_cube.units)

        # Frequency
        if "frequency" in ref_cube.attributes:
            self.attributes["frequency"] = ref_cube.attributes["frequency"]

    @property
    def is_closed(self) -> bool:
        """Check if the file is closed."""
        return self._cubes is None

    def _initialize_entity(self) -> None:
        """Initialize the provenance entity representing the file."""
        super()._initialize_entity()
        settings = {
            "preprocessor:" + k: str(v) for k, v in self.settings.items()
        }
        self.entity.add_attributes(settings)

    def group(self, keys: list) -> str:
        """Generate group keyword.

        Returns a string that identifies a group. Concatenates a list of
        values from .attributes
        """
        if not keys:
            return ""

        if isinstance(keys, str):
            keys = [keys]

        identifier = []
        for key in keys:
            attribute = self.attributes.get(key)
            if attribute:
                if isinstance(attribute, (list, tuple)):
                    attribute = "-".join(attribute)
                identifier.append(attribute)

        return "_".join(identifier)


PreprocessorItem: TypeAlias = PreprocessorFile | Cube | str | Path


def _apply_multimodel(
    products: set[PreprocessorFile],
    step: str,
    debug: bool | None,
) -> set[PreprocessorFile]:
    """Apply multi model step to products."""
    settings, exclude = _get_multi_model_settings(products, step)

    logger.debug(
        "Applying %s to\n%s",
        step,
        "\n".join(str(p) for p in products - exclude),
    )
    result: list[PreprocessorFile] = preprocess(  # type: ignore
        products - exclude,  # type: ignore
        step,
        **settings,
    )
    products = set(result) | exclude

    if debug:
        for product in products:
            logger.debug("Result %s", product.filename)
            if not product.is_closed:
                for cube in product.cubes:
                    logger.debug("with cube %s", cube)

    return products


class PreprocessingTask(BaseTask):
    """Task for running the preprocessor."""

    def __init__(
        self,
        products: Iterable[PreprocessorFile],
        name: str = "",
        order: Iterable[str] = DEFAULT_ORDER,
        debug: bool | None = None,
        write_ncl_interface: bool = False,
    ) -> None:
        """Initialize."""
        _check_multi_model_settings(products)
        super().__init__(name=name, products=products)
        self.order = list(order)
        self.debug = debug
        self.write_ncl_interface = write_ncl_interface

    def _initialize_product_provenance(self) -> None:
        """Initialize product provenance."""
        self._initialize_products(self.products)
        self._initialize_multimodel_provenance()
        self._initialize_ensemble_provenance()

    def _initialize_multiproduct_provenance(self, step: str) -> None:
        input_products = self._get_input_products(step)
        if input_products:
            statistic_products = set()

            for input_product in input_products:
                step_settings = input_product.settings[step]
                output_products = step_settings.get("output_products", {})

                for product in output_products.values():
                    statistic_products.update(product.values())

            self._initialize_products(statistic_products)

    def _initialize_multimodel_provenance(self) -> None:
        """Initialize provenance for multi-model statistics."""
        step = "multi_model_statistics"
        self._initialize_multiproduct_provenance(step)

    def _initialize_ensemble_provenance(self) -> None:
        """Initialize provenance for ensemble statistics."""
        step = "ensemble_statistics"
        self._initialize_multiproduct_provenance(step)

    def _get_input_products(self, step: str) -> list[PreprocessorFile]:
        """Get input products."""
        return [
            product for product in self.products if step in product.settings
        ]

    def _initialize_products(self, products: set[PreprocessorFile]) -> None:
        """Initialize products."""
        for product in products:
            product.initialize_provenance(self.activity)

    def _run(self, _) -> list[str]:  # noqa: C901,PLR0912
        """Run the preprocessor."""
        self._initialize_product_provenance()

        steps = {
            step for product in self.products for step in product.settings
        }
        blocks = get_step_blocks(steps, self.order)

        saved = set()
        delayeds = []
        for block in blocks:
            logger.debug("Running block %s", block)
            if block[0] in MULTI_MODEL_FUNCTIONS:
                for step in block:
                    self.products = _apply_multimodel(
                        self.products,
                        step,
                        self.debug,
                    )
            else:
                for product in _sort_products(self.products):
                    logger.debug("Applying single-model steps to %s", product)
                    for step in block:
                        if step in product.settings:
                            product.apply(step, self.debug)
                    if block == blocks[-1]:
                        product.cubes  # noqa: B018 pylint: disable=pointless-statement
                        delayed = product.close()
                        delayeds.append(delayed)
                        saved.add(product.filename)

        for product in self.products:
            if product.filename not in saved:
                product.cubes  # noqa: B018 pylint: disable=pointless-statement
                delayed = product.close()
                delayeds.append(delayed)

        delayeds = [d for d in delayeds if d is not None]

        if self.scheduler_lock is not None:
            logger.debug("Acquiring save lock for task %s", self.name)
            self.scheduler_lock.acquire()
            logger.debug("Acquired save lock for task %s", self.name)
        try:
            logger.info(
                "Computing and saving data for preprocessing task %s",
                self.name,
            )
            _compute_with_progress(delayeds, description=self.name)
        finally:
            if self.scheduler_lock is not None:
                self.scheduler_lock.release()
                logger.debug("Released save lock for task %s", self.name)

        return write_metadata(
            self.products,
            self.write_ncl_interface,
        )

    def __str__(self) -> str:
        """Get human readable description."""
        order = [
            step
            for step in self.order
            if any(step in product.settings for product in self.products)
        ]
        products = "\n\n".join(
            "\n".join(
                [
                    str(p),
                    "input files: " + pformat(p._input_files),  # noqa: SLF001
                    "settings: " + pformat(p.settings),
                ],
            )
            for p in self.products
        )
        return "\n".join(
            [
                f"{self.__class__.__name__}: {self.name}",
                f"order: {order}",
                f"{products}",
                self.print_ancestors(),
            ],
        )
