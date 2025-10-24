"""Module with functions to check a recipe."""

from __future__ import annotations

import inspect
import logging
import os
import subprocess
from functools import partial
from pprint import pformat
from shutil import which
from typing import TYPE_CHECKING, Any

import isodate
import yamale

import esmvalcore.preprocessor
from esmvalcore.exceptions import InputFilesNotFound, RecipeError
from esmvalcore.local import _get_start_end_year, _parse_period
from esmvalcore.preprocessor import TIME_PREPROCESSORS, PreprocessingTask
from esmvalcore.preprocessor._multimodel import _get_operator_and_kwargs
from esmvalcore.preprocessor._other import _get_var_info
from esmvalcore.preprocessor._regrid import (
    HORIZONTAL_SCHEMES_IRREGULAR,
    HORIZONTAL_SCHEMES_REGULAR,
    HORIZONTAL_SCHEMES_UNSTRUCTURED,
    _load_generic_scheme,
)
from esmvalcore.preprocessor._shared import get_iris_aggregator
from esmvalcore.preprocessor._supplementary_vars import (
    PREPROCESSOR_SUPPLEMENTARIES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from esmvalcore._task import TaskSet
    from esmvalcore.dataset import Dataset
    from esmvalcore.typing import Facets


logger = logging.getLogger(__name__)


def align_metadata(step_settings: dict[str, Any]) -> None:
    """Check settings of preprocessor ``align_metadata``."""
    project = step_settings.get("target_project")
    mip = step_settings.get("target_mip")
    short_name = step_settings.get("target_short_name")
    strict = step_settings.get("strict", True)

    # Any missing arguments will be reported later
    if project is None or mip is None or short_name is None:
        return

    try:
        _get_var_info(project, mip, short_name)
    except ValueError as exc:
        if strict:
            msg = (
                f"align_metadata failed: {exc}. Set `strict=False` to ignore "
                f"this."
            )
            raise RecipeError(msg) from exc
    except KeyError as exc:
        msg = f"align_metadata failed: {exc}"
        raise RecipeError(msg) from exc


def ncl_version() -> None:
    """Check the NCL version."""
    ncl = which("ncl")
    if not ncl:
        msg = (
            "Recipe contains NCL scripts, but cannot find an NCL installation."
        )
        raise RecipeError(msg)
    try:
        cmd = [ncl, "-V"]
        version = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to execute '%s'", " ".join(cmd))
        msg = (
            "Recipe contains NCL scripts, but your NCL "
            "installation appears to be broken."
        )
        raise RecipeError(msg) from exc

    version = version.strip()
    logger.info("Found NCL version %s", version)

    major, minor = (int(i) for i in version.split(".")[:2])
    if major < 6 or (major == 6 and minor < 4):
        msg = (
            "NCL version 6.4 or higher is required to run "
            "a recipe containing NCL scripts."
        )
        raise RecipeError(msg)


def recipe_with_schema(filename: Path) -> None:
    """Check if the recipe content matches schema."""
    schema_file = os.path.join(os.path.dirname(__file__), "recipe_schema.yml")
    logger.debug("Checking recipe against schema %s", schema_file)
    recipe = yamale.make_data(filename)
    schema = yamale.make_schema(schema_file)
    yamale.validate(schema, recipe, strict=False)


def diagnostics(diags: dict[str, dict[str, Any]] | None) -> None:
    """Check diagnostics in recipe."""
    if diags is None:
        msg = "The given recipe does not have any diagnostic."
        raise RecipeError(msg)
    for name, diagnostic in diags.items():
        if "scripts" not in diagnostic:
            msg = f"Missing scripts section in diagnostic '{name}'."
            raise RecipeError(msg)
        variable_names = tuple(diagnostic.get("variables", {}))
        scripts = diagnostic.get("scripts")
        if scripts is None:
            scripts = {}
        for script_name, script in scripts.items():
            if script_name in variable_names:
                msg = (
                    f"Invalid script name '{script_name}' encountered "
                    f"in diagnostic '{name}': scripts cannot have the "
                    "same name as variables."
                )
                raise RecipeError(msg)
            if not script.get("script"):
                msg = (
                    f"No script defined for script '{script_name}' in "
                    f"diagnostic '{name}'."
                )
                raise RecipeError(msg)


def duplicate_datasets(
    datasets: list[dict[str, Any]],
    diagnostic: str,
    variable_group: str,
) -> None:
    """Check for duplicate datasets."""
    if not datasets:
        msg = (
            "You have not specified any dataset or additional_dataset "
            f"groups for variable '{variable_group}' in diagnostic "
            f"'{diagnostic}'."
        )
        raise RecipeError(msg)
    checked_datasets_ = []
    for dataset in datasets:
        if dataset in checked_datasets_:
            msg = (
                f"Duplicate dataset\n{pformat(dataset)}\nfor variable "
                f"'{variable_group}' in diagnostic '{diagnostic}'."
            )
            raise RecipeError(msg)
        checked_datasets_.append(dataset)


def variable(
    var: dict[str, Any],
    required_keys: Iterable[str],
    diagnostic: str,
    variable_group: str,
) -> None:
    """Check variables as derived from recipe."""
    required = set(required_keys)
    missing = required - set(var)
    if missing:
        msg = (
            f"Missing keys {missing} in\n{pformat(var)}\nfor variable "
            f"'{variable_group}' in diagnostic '{diagnostic}'."
        )
        raise RecipeError(msg)


def _log_data_availability_errors(dataset: Dataset) -> None:
    """Check if the required input data is available."""
    input_files = dataset.files
    patterns = dataset._file_globs  # noqa: SLF001
    if not input_files:
        logger.error("No input files found for %s", dataset)
        if patterns:
            if len(patterns) == 1:
                msg = f": {patterns[0]}"
            else:
                msg = "\n{}".format("\n".join(str(p) for p in patterns))
            logger.error("Looked for files matching%s", msg)
        logger.error("Set 'log_level' to 'debug' to get more information")


def _group_years(years: Iterable[int]) -> str:
    """Group an iterable of years into easy to read text.

    Example
    -------
    [1990, 1991, 1992, 1993, 2000] -> "1990-1993, 2000"
    """
    years = sorted(years)
    year = years[0]
    previous_year = year
    starts = [year]
    ends = []
    for year in years[1:]:
        if year != previous_year + 1:
            starts.append(year)
            ends.append(previous_year)
        previous_year = year
    ends.append(year)

    ranges = []
    for start, end in zip(starts, ends, strict=False):
        ranges.append(f"{start}" if start == end else f"{start}-{end}")

    return ", ".join(ranges)


def data_availability(dataset: Dataset, log: bool = True) -> None:
    """Check if input_files cover the required years."""
    input_files = dataset.files
    facets = dataset.facets

    if log:
        _log_data_availability_errors(dataset)

    if not input_files:
        msg = f"Missing data for {dataset.summary(True)}"
        raise InputFilesNotFound(msg)

    if "timerange" not in facets:
        return

    start_date, end_date = _parse_period(facets["timerange"])
    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])
    required_years = set(range(start_year, end_year + 1, 1))
    available_years: set[int] = set()

    for file in input_files:
        start, end = _get_start_end_year(file)
        available_years.update(range(start, end + 1))

    missing_years = required_years - available_years
    if missing_years:
        missing_txt = _group_years(missing_years)

        msg = "No input data available for years {} in files:\n{}".format(
            missing_txt,
            "\n".join(str(f) for f in input_files),
        )
        raise InputFilesNotFound(msg)


def preprocessor_supplementaries(
    dataset: Dataset,
    settings: dict[str, Any],
) -> None:
    """Check that the required supplementary variables have been added."""
    steps = [step for step in settings if step in PREPROCESSOR_SUPPLEMENTARIES]
    supplementaries = {d.facets["short_name"] for d in dataset.supplementaries}

    for step in steps:
        ancs = PREPROCESSOR_SUPPLEMENTARIES[step]
        for short_name in ancs["variables"]:
            if short_name in supplementaries:
                break
        else:
            if ancs["required"] == "require_at_least_one":
                msg = (
                    f"Preprocessor function {step} requires that at least "
                    f"one supplementary variable of {ancs['variables']} is "
                    f"defined in the recipe for {dataset}."
                )
                raise RecipeError(msg)
            if ancs["required"] == "prefer_at_least_one":
                logger.warning(
                    "Preprocessor function %s works best when at least "
                    "one supplementary variable of %s is defined in the "
                    "recipe for %s.",
                    step,
                    ancs["variables"],
                    dataset,
                )


def tasks_valid(tasks: TaskSet) -> None:
    """Check that tasks are consistent."""
    filenames = set()
    msg = "Duplicate preprocessor filename {}, please file a bug report."
    for task in tasks.flatten():
        if isinstance(task, PreprocessingTask):
            for product in task.products:
                if product.filename in filenames:
                    raise ValueError(msg.format(product.filename))
                filenames.add(product.filename)


def check_for_temporal_preprocs(profile: dict[str, Any]) -> None:
    """Check for temporal operations on fx variables."""
    temp_preprocs = [
        preproc
        for preproc in profile
        if profile[preproc] and preproc in TIME_PREPROCESSORS
    ]
    if temp_preprocs:
        msg = (
            f"Time coordinate preprocessor step(s) {temp_preprocs} not permitted on fx "
            "vars, please remove them from recipe"
        )
        raise RecipeError(msg)


def extract_shape(settings: dict[str, Any]) -> None:
    """Check that `extract_shape` arguments are valid."""
    shapefile = settings.get("shapefile", "")
    if not os.path.exists(shapefile):
        msg = (
            f"In preprocessor function `extract_shape`: Unable to find "
            f"'shapefile: {shapefile}'"
        )
        raise RecipeError(msg)

    valid: dict[str, set[Any]] = {
        "method": {"contains", "representative"},
        "crop": {True, False},
        "decomposed": {True, False},
    }
    for key, valid_values in valid.items():
        value = settings.get(key)
        if not (value is None or value in valid_values):
            msg = (
                f"In preprocessor function `extract_shape`: Invalid value "
                f"'{value}' for argument '{key}', choose from "
                "{}".format(", ".join(f"'{k}'".lower() for k in valid_values))
            )
            raise RecipeError(msg)


def _verify_span_value(span: str) -> None:
    """Raise error if span argument cannot be verified."""
    valid_names = ("overlap", "full")
    if span not in valid_names:
        msg = (
            "Invalid value encountered for `span` in preprocessor "
            f"`multi_model_statistics`. Valid values are {valid_names}."
            f"Got {span}."
        )
        raise RecipeError(msg)


def _verify_groupby(groupby: Any) -> None:
    """Raise error if groupby arguments cannot be verified."""
    if not isinstance(groupby, list):
        msg = (
            "Invalid value encountered for `groupby` in preprocessor "
            "`multi_model_statistics`.`groupby` must be defined as a "
            f"list. Got {groupby}."
        )
        raise RecipeError(msg)


def _verify_keep_input_datasets(keep_input_datasets: Any) -> None:
    if not isinstance(keep_input_datasets, bool):
        msg = (
            f"Invalid value encountered for `keep_input_datasets`."
            f"Must be defined as a boolean (true or false). "
            f"Got {keep_input_datasets}."
        )
        raise RecipeError(msg)


def _verify_ignore_scalar_coords(ignore_scalar_coords: Any) -> None:
    if not isinstance(ignore_scalar_coords, bool):
        msg = (
            f"Invalid value encountered for `ignore_scalar_coords`."
            f"Must be defined as a boolean (true or false). Got "
            f"{ignore_scalar_coords}."
        )
        raise RecipeError(msg)


def multimodel_statistics_preproc(settings: dict[str, Any]) -> None:
    """Check that the multi-model settings are valid."""
    span = settings.get("span")  # optional, default: overlap
    if span:
        _verify_span_value(span)

    groupby = settings.get("groupby")  # optional, default: None
    if groupby:
        _verify_groupby(groupby)

    keep_input_datasets = settings.get("keep_input_datasets", True)
    _verify_keep_input_datasets(keep_input_datasets)

    ignore_scalar_coords = settings.get("ignore_scalar_coords", False)
    _verify_ignore_scalar_coords(ignore_scalar_coords)


def ensemble_statistics_preproc(settings: dict[str, Any]) -> None:
    """Check that the ensemble settings are valid."""
    span = settings.get("span", "overlap")  # optional, default: overlap
    if span:
        _verify_span_value(span)

    ignore_scalar_coords = settings.get("ignore_scalar_coords", False)
    _verify_ignore_scalar_coords(ignore_scalar_coords)


def _check_delimiter(timerange: Sequence[str]) -> None:
    if len(timerange) != 2:
        msg = (
            "Invalid value encountered for `timerange`. "
            "Valid values must be separated by `/`. "
            f"Got {timerange} instead."
        )
        raise RecipeError(msg)


def _check_duration_periods(timerange: list[str]) -> None:
    # isodate duration must always start with P
    if timerange[0].startswith("P") and timerange[1].startswith("P"):
        msg = (
            "Invalid value encountered for `timerange`. "
            "Cannot set both the beginning and the end "
            "as duration periods."
        )
        raise RecipeError(msg)

    if timerange[0].startswith("P"):
        try:
            isodate.parse_duration(timerange[0])
        except isodate.isoerror.ISO8601Error as exc:
            msg = (
                f"Invalid value encountered for `timerange`. {timerange[0]} is "
                f"not valid duration according to ISO 8601.\n{exc}"
            )
            raise RecipeError(msg) from exc
    elif timerange[1].startswith("P"):
        try:
            isodate.parse_duration(timerange[1])
        except isodate.isoerror.ISO8601Error as exc:
            msg = (
                f"Invalid value encountered for `timerange`. {timerange[1]} is "
                f"not valid duration according to ISO 8601.\n{exc}"
            )
            raise RecipeError(msg) from exc


def _format_years(date: str) -> str:
    if date != "*" and not date.startswith("P"):
        if len(date) < 4:
            date = date.zfill(4)
    return date


def _check_timerange_values(date: str, timerange: Iterable[str]) -> None:
    # Wildcards are fine
    if date == "*":
        return
    # P must always be in a duration string
    # if T in date, that is a datetime; otherwise it's date
    try:
        if date.startswith("P"):
            isodate.parse_duration(date)
        elif "T" in date:
            isodate.parse_datetime(date)
        else:
            isodate.parse_date(date)
    except isodate.isoerror.ISO8601Error as exc:
        msg = (
            "Invalid value encountered for `timerange`. "
            "Valid value must follow ISO 8601 standard "
            "for dates and duration periods, or be "
            "set to '*' to load available years. "
            f"Got {timerange} instead.\n{exc}"
        )
        raise RecipeError(msg) from exc


def valid_time_selection(timerange: str) -> None:
    """Check that `timerange` tag is well defined."""
    if timerange != "*":
        timerange_list: list[str] = timerange.split("/")
        _check_delimiter(timerange_list)
        _check_duration_periods(timerange_list)
        for date in timerange_list:
            _check_timerange_values(_format_years(date), timerange_list)


def differing_timeranges(
    timeranges: set[str],
    required_vars: list[Facets],
) -> None:
    """Log error if required variables have differing timeranges."""
    if len(timeranges) > 1:
        msg = (
            f"Differing timeranges with values {timeranges} "
            f"found for required variables {required_vars}. "
            "Set `timerange` to a common value."
        )
        raise ValueError(msg)


def _check_literal(
    settings: dict,
    *,
    step: str,
    option: str,
    allowed_values: tuple[None | str, ...],
) -> None:
    """Check that an option for a preprocessor has a valid value."""
    if step not in settings:
        return
    user_value = settings[step].get(option, allowed_values[0])
    if user_value not in allowed_values:
        msg = (
            f"Expected one of {allowed_values} for option `{option}` of "
            f"preprocessor `{step}`, got '{user_value}'"
        )
        raise RecipeError(msg)


bias_type = partial(
    _check_literal,
    step="bias",
    option="bias_type",
    allowed_values=("absolute", "relative"),
)


metric_type = partial(
    _check_literal,
    step="distance_metric",
    option="metric",
    allowed_values=(
        "rmse",
        "weighted_rmse",
        "pearsonr",
        "weighted_pearsonr",
        "emd",
        "weighted_emd",
    ),
)


resample_hours = partial(
    _check_literal,
    step="resample_hours",
    option="interpolate",
    allowed_values=(None, "nearest", "linear"),
)


def _check_ref_attributes(products: set, *, step: str, attr_name: str) -> None:
    """Check that exactly one reference dataset is given."""
    products = {p for p in products if step in p.settings}
    if not products:
        return

    # It is fine to have multiple references when preprocessors are used that
    # combine datasets
    multi_dataset_preprocs = (
        "multi_model_statistics",
        "ensemble_statistics",
    )
    for preproc in multi_dataset_preprocs:
        if any(preproc in p.settings for p in products):
            return

    # Check that exactly one dataset contains the specified facet
    reference_products = []
    for product in products:
        if product.attributes.get(attr_name, False):
            reference_products.append(product)
    if len(reference_products) != 1:
        products_str = [p.filename for p in products]
        if not reference_products:
            ref_products_str = ". "
        else:
            ref_products_str = (
                f":\n{pformat([p.filename for p in reference_products])}.\n"
            )
        msg = (
            f"Expected exactly 1 dataset with '{attr_name}: true' in "
            f"products\n{pformat(products_str)},\nfound "
            f"{len(reference_products):d}{ref_products_str}Please also "
            f"ensure that the reference dataset is not excluded with the "
            f"'exclude' option"
        )
        raise RecipeError(msg)


reference_for_bias_preproc = partial(
    _check_ref_attributes,
    step="bias",
    attr_name="reference_for_bias",
)


reference_for_distance_metric_preproc = partial(
    _check_ref_attributes,
    step="distance_metric",
    attr_name="reference_for_metric",
)


def statistics_preprocessors(settings: dict) -> None:
    """Check options of statistics preprocessors."""
    mm_stats = (
        "multi_model_statistics",
        "ensemble_statistics",
    )
    for step, step_settings in settings.items():
        # For multi-model statistics, we need to check each entry of statistics
        if step in mm_stats:
            _check_mm_stat(step, step_settings)

        # For other statistics, check optional kwargs for operator
        elif "_statistics" in step:
            _check_regular_stat(step, step_settings)


def _check_regular_stat(step: str, step_settings: dict[str, Any]) -> None:
    """Check regular statistics (non-multi-model statistics) step."""
    step_settings = dict(step_settings)

    # Some preprocessors like climate_statistics use default 'mean' for
    # operator. If 'operator' is missing for those preprocessors with no
    # default, this will be detected in PreprocessorFile.check() later.
    operator = step_settings.pop("operator", "mean")

    # If preprocessor does not exist, do nothing here; this will be detected in
    # PreprocessorFile.check() later.
    try:
        preproc_func = getattr(esmvalcore.preprocessor, step)
    except AttributeError:
        return

    # Ignore other preprocessor arguments, e.g., 'hours' for hourly_statistics
    other_args = [
        n
        for (n, p) in inspect.signature(preproc_func).parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ][1:]
    operator_kwargs = {
        k: v for (k, v) in step_settings.items() if k not in other_args
    }
    try:
        get_iris_aggregator(operator, **operator_kwargs)
    except ValueError as exc:
        msg = f"Invalid options for {step}: {exc}"
        raise RecipeError(msg) from exc


def _check_mm_stat(step: str, step_settings: dict[str, Any]) -> None:
    """Check multi-model statistic step."""
    statistics = step_settings.get("statistics", [])
    for stat in statistics:
        try:
            (operator, kwargs) = _get_operator_and_kwargs(stat)
        except ValueError as exc:
            raise RecipeError(str(exc)) from exc
        try:
            get_iris_aggregator(operator, **kwargs)
        except ValueError as exc:
            msg = f"Invalid options for {step}: {exc}"
            raise RecipeError(msg) from exc


def regridding_schemes(settings: dict[str, Any]) -> None:
    """Check :obj:`str` regridding schemes."""
    if "regrid" not in settings:
        return

    # Note: If 'scheme' is missing, this will be detected in
    # PreprocessorFile.check() later
    scheme = settings["regrid"].get("scheme")

    # Check built-in regridding schemes (given as str)
    if isinstance(scheme, str):
        scheme = settings["regrid"]["scheme"]

        allowed_regridding_schemes = list(
            set(
                list(HORIZONTAL_SCHEMES_IRREGULAR)
                + list(HORIZONTAL_SCHEMES_REGULAR)
                + list(HORIZONTAL_SCHEMES_UNSTRUCTURED),
            ),
        )
        if scheme not in allowed_regridding_schemes:
            msg = (
                f"Got invalid built-in regridding scheme '{scheme}', expected "
                f"one of {allowed_regridding_schemes} or a generic scheme "
                f"(see https://docs.esmvaltool.org/projects/ESMValCore/en/"
                f"latest/recipe/preprocessor.html#generic-regridding-schemes)."
            )
            raise RecipeError(msg)

    # Check generic regridding schemes (given as dict)
    if isinstance(scheme, dict):
        try:
            _load_generic_scheme(scheme)
        except ValueError as exc:
            msg = (
                f"Failed to load generic regridding scheme: {exc!s} See "
                f"https://docs.esmvaltool.org/projects/ESMValCore/en/latest"
                f"/recipe/preprocessor.html#generic-regridding-schemes for "
                f"details."
            )
            raise RecipeError(msg) from exc
