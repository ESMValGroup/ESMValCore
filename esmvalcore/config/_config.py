"""Functions dealing with config-user.yml / config-developer.yml."""
from __future__ import annotations

import collections.abc
import fnmatch
import logging
import os
from functools import lru_cache
from importlib.resources import files as importlib_files
from pathlib import Path

import yaml

from esmvalcore.cmor.table import CMOR_TABLES, read_cmor_tables
from esmvalcore.exceptions import RecipeError
from esmvalcore.typing import FacetValue

logger = logging.getLogger(__name__)

TASKSEP = os.sep

CFG = {}


def _deep_update(dictionary, update):
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping):
            dictionary[key] = _deep_update(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value
    return dictionary


@lru_cache()
def _load_extra_facets(project, extra_facets_dir):
    config = {}
    config_paths = [
        importlib_files("esmvalcore.config") / "extra_facets",
        Path.home() / ".esmvaltool" / "extra_facets",
    ]
    config_paths.extend([Path(p) for p in extra_facets_dir])
    for config_path in config_paths:
        config_file_paths = config_path.glob(f"{project.lower()}-*.yml")
        for config_file_path in sorted(config_file_paths):
            logger.debug("Loading extra facets from %s", config_file_path)
            with config_file_path.open(encoding='utf-8') as config_file:
                config_piece = yaml.safe_load(config_file)
            if config_piece:
                _deep_update(config, config_piece)
    return config


def get_extra_facets(dataset, extra_facets_dir):
    """Read configuration files with additional variable information."""
    project_details = _load_extra_facets(
        dataset.facets['project'],
        extra_facets_dir,
    )

    def pattern_filter(patterns, name):
        """Get the subset of the list `patterns` that `name` matches.

        Parameters
        ----------
        patterns : :obj:`list` of :obj:`str`
            A list of strings that may contain shell-style wildcards.
        name : str
            A string describing the dataset, mip, or short_name.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The subset of patterns that `name` matches.
        """
        return [pat for pat in patterns if fnmatch.fnmatchcase(name, pat)]

    extra_facets = {}
    for dataset_ in pattern_filter(project_details, dataset['dataset']):
        for mip_ in pattern_filter(project_details[dataset_], dataset['mip']):
            for var in pattern_filter(project_details[dataset_][mip_],
                                      dataset['short_name']):
                facets = project_details[dataset_][mip_][var]
                extra_facets.update(facets)

    return extra_facets


def load_config_developer(cfg_file):
    """Read the developer's configuration file."""
    with open(cfg_file, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    if 'obs4mips' in cfg:
        logger.warning(
            "Correcting capitalization, project 'obs4mips'"
            " should be written as 'obs4MIPs' in %s", cfg_file)
        cfg['obs4MIPs'] = cfg.pop('obs4mips')

    for project, settings in cfg.items():
        for site, drs in settings.get('input_dir', {}).items():
            # Since v2.8, 'version' can be used instead of 'latestversion'
            if isinstance(drs, list):
                drs = [d.replace('{latestversion}', '{version}') for d in drs]
            else:
                drs = drs.replace('{latestversion}', '{version}')
            settings['input_dir'][site] = drs
        CFG[project] = settings

    read_cmor_tables(cfg_file)


def get_project_config(project):
    """Get developer-configuration for project."""
    if project in CFG:
        return CFG[project]
    raise RecipeError(f"Project '{project}' not in config-developer.yml")


def get_institutes(variable):
    """Return the institutes given the dataset name in CMIP6."""
    dataset = variable['dataset']
    project = variable['project']
    try:
        return CMOR_TABLES[project].institutes[dataset]
    except (KeyError, AttributeError):
        return []


def get_activity(variable):
    """Return the activity given the experiment name in CMIP6."""
    project = variable['project']
    try:
        exp = variable['exp']
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
    if 'ignore_warnings' not in project_cfg:
        return None
    if step not in project_cfg['ignore_warnings']:
        return None
    return project_cfg['ignore_warnings'][step]
