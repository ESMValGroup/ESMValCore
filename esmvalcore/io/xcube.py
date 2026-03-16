"""Access data using `xcube <https://xcube.readthedocs.io>`_.

Run the command ``esmvaltool config copy data-xcube-esacci.yml`` to update
your :ref:`configuration <config-data-sources>` to use this module. This will
create a file with the following content in your configuration directory:

.. literalinclude:: ../configurations/data-xcube-esacci.yml
   :language: yaml
   :caption: Contents of ``data-xcube-esacci.yml``

"""

from __future__ import annotations

import copy
import fnmatch
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import xcube.core.store
from fixer import fix

import esmvalcore.io.protocol
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    import iris.cube

    from esmvalcore.typing import Facets, FacetValue


logger = logging.getLogger(__name__)

FREQUENCIES = {
    "P1D": "day",
    "P1M": "mon",
    "P1Y": "yr",
}


@dataclass
class XCubeDataset(esmvalcore.io.protocol.DataElement):
    """A dataset that can be used to load data found using xcube_."""

    name: str
    """A unique name identifying the data."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    store: xcube.core.store.store.DataStore = field(repr=False)
    """The store containing the data."""

    open_params: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parameters to use when opening the data."""

    _attributes: dict[str, Any] | None = field(
        init=False,
        repr=False,
        default=None,
    )

    def __hash__(self) -> int:
        """Return a number uniquely representing the data element."""
        return hash((self.name, self.facets.get("version")))

    def prepare(self) -> None:
        """Prepare the data for access."""
        self.store.preload_data(self.name)

    @property
    def attributes(self) -> dict[str, Any]:
        """Attributes are key-value pairs describing the data."""
        if self._attributes is None:
            msg = (
                "Attributes have not been read yet. Call the `to_iris` method "
                "first to read the attributes from the file."
            )
            raise ValueError(msg)
        return self._attributes

    @attributes.setter
    def attributes(self, value: dict[str, Any]) -> None:
        self._attributes = value

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        :
            The loaded data.
        """
        dataset = self.store.open_data(self.name, **self.open_params)
        dataset = fix(dataset, self.name)
        dataset.attrs["source_file"] = repr(self)

        # Cache the attributes.
        self.attributes = copy.deepcopy(dataset.attrs)
        return dataset_to_iris(dataset)


_DATASETS_LOGGED: set[str] = set()


@dataclass
class XCubeDataSource(esmvalcore.io.protocol.DataSource):
    """Data source for finding files on a local filesystem."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    data_store_id: str
    """Name of the data store.

    A list of available data stores can be found in the `xcube documentation
    <https://xcube.readthedocs.io/en/latest/dataaccess.html#available-data-stores>`__.
    """

    values: dict[str, dict[str, str]] = field(default_factory=dict)
    """Mapping between the ESMValCore and xcube facet values."""

    data_store_params: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parameters to use when creating the data store."""

    open_params: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parameters to use when opening the data."""

    def find_data(self, **facets: FacetValue) -> list[XCubeDataset]:
        """Find data.

        Parameters
        ----------
        **facets :
            Find data matching these facets.

        Returns
        -------
        :
            A list of data elements that have been found.
        """
        store = xcube.core.store.new_data_store(
            self.data_store_id,
            **self.data_store_params,
        )
        result = []
        requested_short_names = facets.get("short_name", "*")
        if isinstance(requested_short_names, str | int | float):
            requested_short_names = [str(requested_short_names)]
        requested_xcube_short_names = [
            self.values.get("short_name", {}).get(short_name, short_name)
            for short_name in requested_short_names
        ]
        requested_datasets = facets.get("dataset", "*")
        if isinstance(requested_datasets, str | int | float):
            requested_datasets = [str(requested_datasets)]
        available_datasets = store.list_data_ids()

        self.debug_info = (
            "No dataset matching "
            + ", ".join(f"'{d}'" for d in requested_datasets)
            + f" was found in {self.data_store_id}. Available datasets are:\n"
            + "\n".join(sorted(available_datasets))
        )
        for data_id in available_datasets:
            for dataset_pattern in requested_datasets:
                if fnmatch.fnmatchcase(data_id, dataset_pattern):
                    description = store.describe_data(data_id)
                    available_xcube_short_names = list(description.data_vars)
                    xcube_short_names = [
                        short_name
                        for short_name in available_xcube_short_names
                        for short_name_pattern in requested_xcube_short_names
                        if fnmatch.fnmatchcase(short_name, short_name_pattern)
                    ]
                    if not xcube_short_names:
                        self.debug_info = (
                            "No variable matching "
                            + ", ".join(
                                f"'{s}'" for s in requested_xcube_short_names
                            )
                            + f" was found in dataset '{data_id}'. Available variables are:\n"
                            + "\n".join(sorted(available_xcube_short_names))
                        )
                        continue

                    timerange = f"{description.time_range[0]}/{description.time_range[1]}".replace(
                        "-",
                        "",
                    )
                    short_names = [
                        short_name
                        for short_name, xcube_short_name in self.values.get(
                            "short_name",
                            {},
                        ).items()
                        if xcube_short_name in xcube_short_names
                    ]
                    dataset = XCubeDataset(
                        name=data_id,
                        facets={
                            "dataset": data_id,
                            "short_name": (
                                short_names[0]
                                if len(short_names) == 1
                                else short_names
                            ),
                            "timerange": timerange,
                        },
                        store=store,
                        open_params=copy.deepcopy(self.open_params),
                    )
                    frequency = FREQUENCIES.get(
                        description.attrs.get("time_coverage_resolution", ""),
                    )
                    if frequency:
                        # Assign the frequency facet if it is a known frequency.
                        dataset.facets["frequency"] = frequency
                    dataset.attributes = description.attrs

                    result.append(dataset)

        if result:
            self.debug_info = (
                f"Found dataset{'' if len(result) == 1 else 's'} "
                f"{', '.join(d.name for d in result)} in data store "
                f"{self.data_store_id}."
            )

        return result
