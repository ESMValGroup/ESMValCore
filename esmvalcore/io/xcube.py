"""Access data using `xcube <https://xcube.readthedocs.io>`_.

Run the command ``esmvaltool config copy data-xcube-ccizarr.yml`` to update
your :ref:`configuration <config-data-sources>` to use this module. This will
create a file with the following content in your configuration directory:

.. literalinclude:: ../configurations/data-xcube-ccizarr.yml
   :language: yaml
   :caption: Contents of ``data-xcube-ccizarr.yml``

"""

from __future__ import annotations

import copy
import fnmatch
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import iris.cube
import iris.std_names
import xcube.core.store

import esmvalcore.io.protocol
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    from esmvalcore.typing import Facets, FacetValue


logger = logging.getLogger(__name__)

FREQUENCIES = {
    "P1M": "mon",
    "P1D": "day",
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
        # Keep only variables matching the "short_name" facet.
        short_names = self.facets.get("short_name", [])
        if isinstance(short_names, str | int):
            short_names = [str(short_names)]
        if short_names:
            dataset = dataset[short_names]

        # Drop invalid standard_names.
        # TODO: move this to a standalone fixes package.
        for data_var in dataset.data_vars.values():
            if (
                "standard_name" in data_var.attrs
                and data_var.attrs["standard_name"]
                not in iris.std_names.STD_NAMES
            ):
                data_var.attrs.pop("standard_name")

        # Cache the attributes.
        self.attributes = copy.deepcopy(dataset.attrs)
        return dataset_to_iris(dataset)


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

    data_store_params: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parameters to use when creating the data store."""

    open_params: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parameters to use when opening the data."""

    def find_data(self, **facets: FacetValue) -> list[XCubeDataset]:  # noqa: C901,PLR0912
        # TODO: fix complexity
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
        requested_datasets = facets.get("dataset", "*")
        if isinstance(requested_datasets, str | int | float):
            requested_datasets = [str(requested_datasets)]
        available_datasets = store.list_data_ids()

        for data_id in available_datasets:
            for dataset_pattern in requested_datasets:
                if fnmatch.fnmatchcase(data_id, dataset_pattern):
                    description = store.describe_data(data_id)
                    available_short_names = list(description.data_vars)
                    short_names = [
                        short_name
                        for short_name in available_short_names
                        for short_name_pattern in requested_short_names
                        if fnmatch.fnmatchcase(short_name, short_name_pattern)
                    ]
                    if not short_names:
                        logger.debug(
                            "No variable matching % found in %s. Available variables are: %s",
                            requested_short_names,
                            data_id,
                            available_short_names,
                        )
                    # TODO: Maybe this is too complicated and we should only
                    # decide which variables to keep/drop after load and conversion
                    # to iris cube.
                    open_params = copy.deepcopy(self.open_params)
                    open_params_schema = store.get_open_data_params_schema()
                    if "variable_names" in open_params_schema.properties:
                        open_params["variable_names"] = short_names
                    elif "drop_variables" in open_params_schema.properties:
                        drop_variables = {
                            short_name
                            for short_name in available_short_names
                            if short_name not in short_names
                        }
                        for coord in description.coords.values():
                            if bound_var := coord.attrs.get("bounds"):
                                drop_variables.remove(bound_var)
                        for data_var in description.data_vars.values():
                            # TODO: keep cell measures
                            for ancillary_var in data_var.attrs.get(
                                "ancillary_variables",
                                "",
                            ).split():
                                drop_variables.remove(ancillary_var)
                        open_params["drop_variables"] = sorted(drop_variables)
                    timerange = f"{description.time_range[0]}/{description.time_range[1]}".replace(
                        "-",
                        "",
                    )
                    native_frequency = description.attrs[
                        "time_coverage_resolution"
                    ]
                    frequency = FREQUENCIES.get(
                        native_frequency,
                        native_frequency,
                    )

                    dataset = XCubeDataset(
                        name=data_id,
                        facets={
                            "dataset": data_id,
                            "short_name": short_names
                            if len(short_names) > 1
                            else short_names[0],
                            "frequency": frequency,
                            "timerange": timerange,
                        },
                        store=store,
                        open_params=open_params,
                    )
                    dataset.attributes = description.attrs

                    result.append(dataset)

        if not result:
            logger.debug(
                "No datasets matching %s found in %s. Available datasets are: %s",
                requested_datasets,
                self.data_store_id,
                available_datasets,
            )

        return result
