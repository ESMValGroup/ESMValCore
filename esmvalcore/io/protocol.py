"""Protocols for accessing data.

An input data source can be defined in the configuration by using :obj:`esmvalcore.config.CFG`

.. code-block:: python

    >>> from esmvalcore.config import CFG
    >>> CFG["projects"]["example-project"]["data"]["example-source-name"] = {
            "type": "example_module.ExampleDataSource"
            "argument1": "value1"
            "argument2": "value2"
        }

or as a :ref:`YAML configuration file <config_overview>`

.. code-block:: yaml

    projects:
      example-project:
        data:
          example-source-name
            type: example_module.ExampleDataSource
            argument1: value1
            argument2: value2


where ``example-project`` is a project, e.g. ``CMIP6``, and ``example-source-name``
is a unique name describing the data source. The datasource type, in the
example above called ``example_module.ExampleDataSource`` needs to implement the
:class:`esmvalcore.io.protocol.DataSource` protocol. Any remaining key-value pairs
in the configuration, ``argument1: value1`` and ``argument2: value2`` are
passed as keyword arguments to the data source when it is created.

By default, all data sources are created with ``priority: 1``. Data sources
with a lower ``priority`` value are searched first and if two data sources
provide the same data, the data from the source with the lowest ``priority`` value
will be used.
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import iris.cube

from esmvalcore.typing import FacetValue


@runtime_checkable
class DataElement(Protocol):
    """A data element represents some data that can be loaded.

    A file is an example of a data element.
    """

    name: str
    """A unique name identifying the data."""

    facets: dict[str, FacetValue]
    """Facets are key-value pairs that can be used for searching the data."""

    attributes: dict[str, Any]
    """Attributes are key-value pairs describing the data."""

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
