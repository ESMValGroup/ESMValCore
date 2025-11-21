"""A modular system for reading input data from various sources.

An input data source can be defined in the configuration by using
:obj:`esmvalcore.config.CFG`, for example:

.. code-block:: python

    >>> from esmvalcore.config import CFG
    >>> CFG["projects"]["CMIP6"]["data"]["local"] = {
            "type": "esmvalcore.local.LocalDataSource",
            "rootpath": "~/climate_data",
            "dirname_template": "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}",
            "filename_template": "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc",
        }

or as a :ref:`YAML configuration file <config_overview>`:

.. code-block:: yaml

    projects:
      CMIP6:
        data:
          local:
            type: "esmvalcore.local.LocalDataSource"
            rootpath: "~/climate_data"
            dirname_template: "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}"
            filename_template: "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc"

where ``CMIP6`` is a project, and ``local`` is a unique name describing the
data source. The data source type,
:class:`esmvalcore.local.LocalDataSource`, in the example above, needs to
implement the :class:`esmvalcore.io.protocol.DataSource` protocol. Any
remaining key-value pairs in the configuration, ``rootpath``,
``dirname_template``, and ``filename_template`` in this example, are passed
as keyword arguments to the data source when it is created.

If there are multiple data sources configured for a project, deduplication of
search results happens based on the
:attr:`esmvalcore.io.protocol.DataElement.name` attribute and the ``"version"``
facet in :attr:`esmvalcore.io.protocol.DataElement.facets` of the data elements
provided by the data sources. If no ``version`` facet is specified in the
search, the latest version will be used. If there is a tie, the data element
provided by the data source with the lowest value of
:attr:`esmvalcore.io.protocol.DataSource.priority` is chosen.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import esmvalcore.io.protocol

if TYPE_CHECKING:
    from esmvalcore.config import Session

logger = logging.getLogger(__name__)


def load_data_sources(
    session: Session,
    project: str | None = None,
) -> list[esmvalcore.io.protocol.DataSource]:
    """Get the list of available data sources.

    If no ``priority`` is configured for a data source, the default priority
    of 1 is used.

    Arguments
    ---------
    session:
        The configuration.
    project:
        If specified, only data sources for this project are returned.

    Returns
    -------
    :
        A list of available data sources.

    Raises
    ------
    ValueError:
        If the project or its settings are not found in the configuration.

    """
    data_sources: list[esmvalcore.io.protocol.DataSource] = []
    if project is not None and project not in session["projects"]:
        msg = f"Unknown project '{project}', please configure it under 'projects'."
        raise ValueError(msg)
    settings = (
        session["projects"]
        if project is None
        else {project: session["projects"][project]}
    )
    for project_, project_settings in settings.items():
        for name, orig_kwargs in project_settings.get("data", {}).items():
            kwargs = orig_kwargs.copy()
            module_name, cls_name = kwargs.pop("type").rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            priority = kwargs.pop("priority", 1)
            data_source = cls(
                name=name,
                project=project_,
                priority=priority,
                **kwargs,
            )
            if not isinstance(data_source, esmvalcore.io.protocol.DataSource):
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
        raise ValueError(msg)
    return data_sources
