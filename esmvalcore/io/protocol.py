"""Protocols for accessing data.

This module defines the :class:`DataSource` and :class:`DataElement` protocols
for finding and loading data. A data source can be used to find data elements
matching specific facets. A data element represents some data that can be
loaded as Iris cubes.

To add support for a new data source, write two classes that implement these
protocols and configure the tool to use the newly implemented data source as
described in :mod:`esmvalcore.io`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

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

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

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
