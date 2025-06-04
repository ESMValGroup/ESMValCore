"""Functions dealing with config-developer.yml and extra facets."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

import iris
import yaml

from esmvalcore.cmor.table import CMOR_TABLES, read_cmor_tables
from esmvalcore.exceptions import RecipeError

if TYPE_CHECKING:
    from esmvalcore.typing import FacetValue

logger = logging.getLogger(__name__)

TASKSEP = os.sep

CFG = {}


# Set iris.FUTURE flags
for attr, value in {
    "save_split_attrs": True,
    "date_microseconds": True,
}.items():
    with contextlib.suppress(AttributeError):
        setattr(iris.FUTURE, attr, value)


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
                drs = [d.replace("{latestversion}", "{version}") for d in drs]
            else:
                drs = drs.replace("{latestversion}", "{version}")
            settings["input_dir"][site] = drs
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
