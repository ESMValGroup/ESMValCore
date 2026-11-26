"""Module for configuring data sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml

from esmvalcore.exceptions import InvalidConfigParameter
from esmvalcore.io import load_data_sources

if TYPE_CHECKING:
    from esmvalcore.config import Session
    from esmvalcore.io.protocol import DataSource

logger = logging.getLogger(__name__)


def _get_data_sources(
    session: Session,
    project: str,
) -> list[DataSource]:
    """Get the list of available data sources.

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
    except ValueError as exc:
        cfg_snippet = {
            "projects": {
                project: {
                    "data": session["projects"]
                    .get(project, {})
                    .get("data", {}),
                },
            },
        }
        msg = (
            f"No data sources found for project '{project}'. Current configuration:\n"
            f"{yaml.safe_dump(cfg_snippet)}"
            "Please configure a data source by following the instructions at "
            "https://docs.esmvaltool.org/projects/ESMValCore/en/latest/"
            "quickstart/configure.html#project-specific-configuration"
        )
        raise InvalidConfigParameter(msg) from exc
