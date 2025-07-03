import importlib
import logging

import esmvalcore.esgf
import esmvalcore.local
from esmvalcore.config import Session
from esmvalcore.io.protocol import DataSource

logger = logging.getLogger(__name__)


def get_data_sources(session: Session) -> list[DataSource]:
    data_sources: list[DataSource] = []
    for project, project_settings in session["projects"].items():
        if "data" not in project_settings:
            logger.info("Using legacy data sources for project '%s'", project)
            # Use legacy data sources from config-user.yml.
            legacy_local_sources = esmvalcore.local._get_data_sources(project)
            data_sources.extend(legacy_local_sources)
            if (
                session["search_esgf"] != "never"
                and project in esmvalcore.esgf.facets.FACETS
            ):
                data_source = esmvalcore.esgf.ESGFDataSource(
                    name="legacy",
                    project=project,
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
                project=project,
                priority=priority,
                **kwargs,
            )
            if not isinstance(data_source, DataSource):
                msg = (
                    "Expected a data source of type `esmvalcore.io.protocol.DataSource`, "
                    f"but your configuration for project '{project}' contains "
                    f"'{data_source}' of type '{type(data_source)}'."
                )
                raise TypeError(msg)
            data_sources.append(data_source)
    return data_sources
