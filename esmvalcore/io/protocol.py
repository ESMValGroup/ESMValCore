from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import iris.cube
import xarray as xr

from esmvalcore.typing import FacetValue


@runtime_checkable
class DataElement(Protocol):
    name: str
    """A unique name identifying the data."""

    facets: dict[str, FacetValue]
    """Facets are key-value pairs describing the data."""

    def __hash__(self) -> int:
        return hash(self.name)

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


@runtime_checkable
class Fix(Protocol):
    def __call__(
        self,
        *datasets: xr.Dataset,
        dataset_id: str | None = None,
    ) -> Iterable[xr.Dataset]:
        pass
