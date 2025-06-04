"""Functions dealing with extra facets."""

from __future__ import annotations

import collections.abc
import fnmatch
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from esmvalcore.config._config_object import CFG

if TYPE_CHECKING:
    from esmvalcore.dataset import Dataset

logger = logging.getLogger(__name__)

TASKSEP = os.sep


def _deep_update(dictionary, update):
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping):
            dictionary[key] = _deep_update(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value
    return dictionary


# TODO: remove in v2.15.0
@lru_cache
def _load_extra_facets(project: str, extra_facets_dir: tuple[Path]) -> dict:
    """Load (deprecated) user-defined extra facets."""
    config: dict[str, dict] = {}
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


def get_extra_facets(dataset: Dataset) -> dict:
    """Read files with additional variable information ("extra facets")."""
    extra_facets: dict[str, str] = {}

    def pattern_filter(
        patterns: collections.abc.Iterable[str],
        name: str,
    ) -> list[str]:
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

    raw_extra_facets = CFG["extra_facets"]
    projects = pattern_filter(raw_extra_facets, dataset["project"])
    for project in projects:
        dataset_names = pattern_filter(
            raw_extra_facets[project],
            dataset["dataset"],
        )
        for dataset_name in dataset_names:
            mips = pattern_filter(
                raw_extra_facets[project][dataset_name],
                dataset["mip"],
            )
            for mip in mips:
                variables = pattern_filter(
                    raw_extra_facets[project][dataset_name][mip],
                    dataset["short_name"],
                )
                for var in variables:
                    facets = raw_extra_facets[project][dataset_name][mip][var]
                    extra_facets.update(facets)

    # Add deprecated user-defined extra facets
    # TODO: remove in v2.15.0
    project_details = _load_extra_facets(
        dataset.facets["project"],
        tuple(CFG["extra_facets_dir"]),
    )
    dataset_names = pattern_filter(project_details, dataset["dataset"])
    for dataset_name in dataset_names:
        mips = pattern_filter(project_details[dataset_name], dataset["mip"])
        for mip in mips:
            variables = pattern_filter(
                project_details[dataset_name][mip],
                dataset["short_name"],
            )
            for var in variables:
                facets = project_details[dataset_name][mip][var]
                extra_facets.update(facets)

    return extra_facets
