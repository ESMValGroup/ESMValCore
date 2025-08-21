"""Module for configuring data sources."""

import importlib
import logging

import esmvalcore.esgf
import esmvalcore.local
from esmvalcore.config import Session
from esmvalcore.io.protocol import DataSource

logger = logging.getLogger(__name__)


def _get_data_sources(
    session: Session,
    project: str | None,
) -> list[DataSource]:
    """Get the list of available data sources.

    Arguments
    ---------
    session:
        The configuration.
    project:
        If specified, only data sources for this project are returned.

    Returns
    -------
    :obj:`list` of :obj:`DataSource`:
        A list of available data sources.

    Raises
    ------
    TypeError:
        If a data source in the configuration is not of type `DataSource`.
    KeyError:
        If the project or its settings are not found in the configuration.

    """
    data_sources: list[DataSource] = []
    if project is not None and project not in session["projects"]:
        msg = f"Unknown project '{project}', please configure it under 'projects'."
        raise KeyError(msg)
    settings = (
        session["projects"]
        if project is None
        else {project: session["projects"][project]}
    )
    for project_, project_settings in settings.items():
        if "data" not in project_settings:
            logger.info("Using legacy data sources for project '%s'", project_)
            # Use legacy data sources from config-user.yml.
            legacy_local_sources = esmvalcore.local._get_data_sources(project_)  # noqa: SLF001
            data_sources.extend(legacy_local_sources)
            if (
                session["search_esgf"] != "never"
                and project_ in esmvalcore.esgf.facets.FACETS
            ):
                data_source = esmvalcore.esgf.ESGFDataSource(
                    name="legacy-esgf",
                    project=project_,
                    priority=2,
                    download_dir=session["download_dir"],
                )
                data_sources.append(data_source)
        for name, kwargs in project_settings.get("data", {}).items():
            kwargs = kwargs.copy()
            module_name, cls_name = kwargs.pop("type").rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            if session["search_esgf"] == "never" and isinstance(
                cls,
                esmvalcore.esgf.ESGFDataSource,
            ):
                continue
            priority = kwargs.pop("priority", 1)
            data_source = cls(
                name=name,
                project=project_,
                priority=priority,
                **kwargs,
            )
            if not isinstance(data_source, DataSource):
                msg = (
                    "Expected a data source of type `esmvalcore.io.protocol.DataSource`, "
                    f"but your configuration for project '{project_}' contains "
                    f"'{data_source}' of type '{type(data_source)}'."
                )
                raise TypeError(msg)
            data_sources.append(data_source)

    if not data_sources:
        if project is None:
            msg = "No data sources found. Check your configuration under 'projects'"
        else:
            msg = (
                f"No data sources found for project '{project}'. "
                f"Check your configuration under 'projects: {project}: data'"
            )
        raise KeyError(msg)
    return data_sources
