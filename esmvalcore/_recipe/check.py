"""Check a recipe while its tasks are created.

These checks find out whether a recipe can run on this machine, so they
depend on more than the recipe and the configuration: the input data that was
found, the facets of each dataset after they have been completed, other files
on disk such as shapefiles, or the installed software. Each check raises a
:class:`~esmvalcore.exceptions.RecipeError` for the first problem it finds.

Checks of whether the recipe itself is correct belong in
:mod:`esmvalcore._recipe.validate`, so the problems are found before the data
is searched, and also by editors.
"""

from __future__ import annotations

import logging
import os
import subprocess
from functools import partial
from pprint import pformat
from shutil import which
from typing import TYPE_CHECKING, Any

import isodate

from esmvalcore.exceptions import InputFilesNotFound, RecipeError
from esmvalcore.io.local import _parse_period
from esmvalcore.preprocessor import TIME_PREPROCESSORS, PreprocessingTask
from esmvalcore.preprocessor._other import _get_var_info
from esmvalcore.preprocessor._supplementary_vars import (
    PREPROCESSOR_SUPPLEMENTARIES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from esmvalcore._task import TaskSet
    from esmvalcore.dataset import Dataset
    from esmvalcore.typing import Facets


logger = logging.getLogger(__name__)


def align_metadata(step_settings: dict[str, Any]) -> None:
    """Check settings of preprocessor ``align_metadata``."""
    project = step_settings.get("target_project")
    mip = step_settings.get("target_mip")
    short_name = step_settings.get("target_short_name")
    branding_suffix = step_settings.get("target_branding_suffix")
    strict = step_settings.get("strict", True)

    # Any missing arguments will be reported later
    if project is None or mip is None or short_name is None:
        return

    try:
        _get_var_info(
            project,
            mip,
            short_name,
            branding_suffix=branding_suffix,
        )
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


def get_no_data_message(dataset: Dataset) -> str:
    """Generate a message for debugging missing data in dataset."""
    lines = [
        f"No files were found for {dataset},\nusing data sources:",
        "\n".join(
            f"- data source: {data_source}\n  message: {data_source.debug_info}"
            for data_source in sorted(
                dataset._used_data_sources,  # noqa: SLF001
                key=lambda d: d.priority,
            )
        ),
    ]
    return "\n".join(lines)


def _log_data_availability_errors(dataset: Dataset) -> None:
    """Check if the required input data is available."""
    if not dataset.files:
        msg = get_no_data_message(dataset)
        logger.error(msg)
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

    if (
        "*" in facets.get("timerange", "*")  # type: ignore[operator]
        or any("timerange" not in f.facets for f in input_files)
    ):
        return

    start_date, end_date = _parse_period(facets["timerange"])
    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])
    required_years = set(range(start_year, end_year + 1, 1))
    available_years: set[int] = set()

    for file in input_files:
        start_date, end_date = file.facets["timerange"].split("/")  # type: ignore[union-attr]
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        available_years.update(range(start_year, end_year + 1))

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
