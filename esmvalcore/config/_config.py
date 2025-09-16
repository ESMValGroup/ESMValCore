"""Functions dealing with config-developer.yml and extra facets."""

from __future__ import annotations

import collections.abc
import contextlib
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import iris
import yaml

from esmvalcore.cmor.table import CMOR_TABLES, read_cmor_tables
from esmvalcore.exceptions import ESMValCoreDeprecationWarning, RecipeError

if TYPE_CHECKING:
    from esmvalcore.typing import FacetValue

logger = logging.getLogger(__name__)

TASKSEP = os.sep

CFG = {}

# TODO: remove in v2.15.0
USER_EXTRA_FACETS = Path.home() / ".esmvaltool" / "extra_facets"


# Set iris.FUTURE flags
for attr, value in {
    "save_split_attrs": True,
    "date_microseconds": True,
}.items():
    with contextlib.suppress(AttributeError):
        setattr(iris.FUTURE, attr, value)


# TODO: remove in v2.15.0
def _deep_update(dictionary, update):
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping):
            dictionary[key] = _deep_update(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value
    return dictionary


# TODO: remove in v2.15.0
@lru_cache
def load_extra_facets(
    project: str,
    extra_facets_dir: tuple[Path],
) -> dict[str, dict[str, Any]]:
    """Load deprecated extra facets."""
    warn_if_old_extra_facets_exist()
    config: dict[str, dict[str, Any]] = {}
    config_paths = [Path.home() / ".esmvaltool" / "extra_facets"]
    config_paths.extend([Path(p) for p in extra_facets_dir])
    for config_path in config_paths:
        config_file_paths = config_path.glob(f"{project.lower()}-*.yml")
        for config_file_path in sorted(config_file_paths):
            logger.debug("Loading extra facets from %s", config_file_path)
            with config_file_path.open(encoding="utf-8") as config_file:
                config_piece = yaml.safe_load(config_file)
            if config_piece:
                _deep_update(config, config_piece)
    return config


# TODO: remove in v2.15.0
def warn_if_old_extra_facets_exist() -> None:
    """Warn user if deprecated dask configuration file exists."""
    if USER_EXTRA_FACETS.exists() and not os.environ.get(
        "ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG",
    ):
        deprecation_msg = (
            "Usage of extra facets located in ~/.esmvaltool/extra_facets has "
            "been deprecated in ESMValCore version 2.13.0 and is scheduled "
            "for removal in version 2.15.0. Please use the configuration "
            "option `extra_facets` instead (see "
            "https://github.com/ESMValGroup/ESMValCore/pull/2747 for "
            "details). To silent this warning and ignore deprecated extra "
            "facets, set the environment variable "
            "ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG=1."
        )
        warnings.warn(
            deprecation_msg,
            ESMValCoreDeprecationWarning,
            stacklevel=2,
        )


def load_config_developer(cfg_file):
    """Read the developer's configuration file."""
    with open(cfg_file, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if "obs4mips" in cfg:
        logger.warning(
            "Correcting capitalization, project 'obs4mips'"
            " should be written as 'obs4MIPs' in %s",
            cfg_file,
        )
        cfg["obs4MIPs"] = cfg.pop("obs4mips")

    for project, settings in cfg.items():
        for site, drs in settings.get("input_dir", {}).items():
            # Since v2.8, 'version' can be used instead of 'latestversion'
            if isinstance(drs, list):
                normalized_drs = [
                    d.replace("{latestversion}", "{version}") for d in drs
                ]
            else:
                normalized_drs = drs.replace("{latestversion}", "{version}")
            settings["input_dir"][site] = normalized_drs
        CFG[project] = settings

    read_cmor_tables(cfg_file)


def get_project_config(project):
    """Get developer-configuration for project."""
    if project in CFG:
        return CFG[project]
    msg = f"Project '{project}' not in config-developer.yml"
    raise RecipeError(msg)


def get_institutes(variable):
    """Return the institutes given the dataset name in CMIP6."""
    dataset = variable["dataset"]
    project = variable["project"]
    try:
        return CMOR_TABLES[project].institutes[dataset]
    except (KeyError, AttributeError):
        return []


def get_activity(variable):
    """Return the activity given the experiment name in CMIP6."""
    project = variable["project"]
    try:
        exp = variable["exp"]
        if isinstance(exp, list):
            return [CMOR_TABLES[project].activities[value][0] for value in exp]
        return CMOR_TABLES[project].activities[exp][0]
    except (KeyError, AttributeError):
        return None


def get_ignored_warnings(project: FacetValue, step: str) -> None | list:
    """Get ignored warnings for a given preprocessing step."""
    if project not in CFG:
        return None
    project_cfg = CFG[project]
    if "ignore_warnings" not in project_cfg:
        return None
    if step not in project_cfg["ignore_warnings"]:
        return None
    return project_cfg["ignore_warnings"][step]
