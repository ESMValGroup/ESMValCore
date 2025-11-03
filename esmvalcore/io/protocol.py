"""Protocols for accessing data.

An input data source can be defined in the configuration by using :obj:`esmvalcore.config.CFG`

.. code-block:: python

    >>> from esmvalcore.config import CFG
    >>> CFG["projects"]["CMIP6"]["data"]["local"] = {
            "type": "esmvalcore.local.LocalDataSource",
            "rootpath": "~/climate_data",
            "dirname_template": "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}",
            "filename_template": "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc",
        }

or as a :ref:`YAML configuration file <config_overview>`

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

Deduplication of search results happens based on the
:attr:`esmvalcore.io.protocol.DataElement.name` attribute and the ``"version"``
facet in :attr:`esmvalcore.io.protocol.DataElement.facets` of the data elements
provided by the data sources. If there is a tie, the data element provided by
the data source with the lowest value of
:attr:`esmvalcore.io.protocol.DataSource.priority` is chosen.
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import iris.cube

from esmvalcore.typing import FacetValue


@runtime_checkable
class DataElement(Protocol):
    """A data element represents some data that can be loaded.

    An :class:`esmvalcore.local.LocalFile` is an example of a data element.
    """

    name: str
    """A unique name identifying the data."""

    facets: dict[str, FacetValue]
    """Facets are key-value pairs that can be used for searching the data."""

    attributes: dict[str, Any]
    """Attributes are key-value pairs describing the data."""

    def __hash__(self) -> int:
        """Return a number uniquely representing the data element."""

    def prepare(self) -> None:
        """Prepare the data for access."""

    def to_iris(
        self,
        ignore_warnings: list[dict[str, Any]] | None = None,
    ) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Parameters
        ----------
        ignore_warnings:
            Keyword arguments passed to :func:`warnings.filterwarnings` used to
            ignore warnings issued by :func:`iris.load_raw`. Each list element
            corresponds to one call to :func:`warnings.filterwarnings`.

        Returns
        -------
        iris.cube.CubeList
            The loaded data.
        """


@runtime_checkable
class DataSource(Protocol):
    """A data source can be used to find data."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    debug_info: str
    """A string containing debug information when no data is found."""

    def find_data(self, **facets: FacetValue) -> Iterable[DataElement]:
        """Find data.

        Parameters
        ----------
        **facets :
            Find data matching these facets.

        Returns
        -------
        :obj:`typing.Iterable` of :obj:`esmvalcore.io.base.DataElement`
            The data elements that have been found.
        """
