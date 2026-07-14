"""Functions dealing with config-developer.yml and extra facets."""
# TODO: remove this module in v2.16.0

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import yaml

from esmvalcore.cmor.table import read_cmor_tables
from esmvalcore.exceptions import RecipeError

if TYPE_CHECKING:
    from pathlib import Path

    from esmvalcore.typing import FacetValue

logger = logging.getLogger(__name__)

TASKSEP = os.sep

CFG: dict[str, Any] = {}


def load_config_developer(
    cfg_file: Path,
    set_cmor_tables: bool = True,
) -> dict:
    """Read the developer's configuration file."""
    with open(cfg_file, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    for lower_case_project in ("obs4mips", "ana4mips"):
        if lower_case_project in cfg:
            project = f"{lower_case_project[:3]}4MIPs"
            logger.warning(
                "Correcting capitalization, project '%s' should be written as '%s' in %s",
                lower_case_project,
                project,
                cfg_file,
            )
            cfg[project] = cfg.pop(lower_case_project)

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

    if set_cmor_tables:
        read_cmor_tables(cfg_file)
    return cfg


def get_project_config(project):
    """Get developer-configuration for project."""
    if project in CFG:
        return CFG[project]
    msg = f"Project '{project}' not in config-developer.yml"
    raise RecipeError(msg)


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
