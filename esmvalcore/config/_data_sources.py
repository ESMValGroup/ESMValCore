"""Module for configuring data sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml

import esmvalcore.esgf
import esmvalcore.esgf.facets
import esmvalcore.local
from esmvalcore.exceptions import InvalidConfigParameter, RecipeError
from esmvalcore.io import load_data_sources

if TYPE_CHECKING:
    from esmvalcore.config import Session
    from esmvalcore.io.protocol import DataSource

logger = logging.getLogger(__name__)


def _get_data_sources(
    session: Session,
    project: str,
) -> list[DataSource]:
    """Get the list of available data sources including legacy configuration.

    Arguments
    ---------
    session:
        The configuration.
    project:
        Data sources for this project are returned.

    Returns
    -------
    :obj:`list` of :obj:`DataSource`:
        A list of available data sources.

    Raises
    ------
    InvalidConfigParameter:
        If the project or its settings are not found in the configuration.

    """
    try:
        return load_data_sources(session, project)
    except ValueError:
        pass

    # Use legacy data sources from config-user.yml and config-developer.yml.
    data_sources: list[DataSource] = []
    try:
        legacy_local_data_sources = esmvalcore.local._get_data_sources(project)  # noqa: SLF001
    except (RecipeError, KeyError):
        # The project is not configured in config-developer.yml
        legacy_local_data_sources = []
    else:
        if (
            session.get("search_esgf", "") != "never"
            and project in esmvalcore.esgf.facets.FACETS
        ):
            data_source = esmvalcore.esgf.ESGFDataSource(
                name="legacy-esgf",
                project=project,
                priority=2,
                download_dir=session["download_dir"],
            )
            data_sources.append(data_source)
    data_sources.extend(legacy_local_data_sources)

    if not data_sources:
        cfg_snippet = {
            "projects": {
                p: {
                    "data": session["projects"].get(p, {}).get("data", {}),
                }
                for p in (
                    session["projects"] if project is None else [project]
                )
            },
        }
        msg = (
            f"No data sources found for project '{project}'. Current configuration:\n"
            f"{yaml.safe_dump(cfg_snippet)}"
            "Please configure a data source by following the instructions at "
            "https://docs.esmvaltool.org/projects/ESMValCore/en/latest/"
            "quickstart/configure.html#project-specific-configuration"
        )
        raise InvalidConfigParameter(msg)
    return data_sources
