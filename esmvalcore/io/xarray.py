"""Access data using :meth:`xarray.open_dataset`.

To use this module, copy one of the following
:ref:`configuration <config-data-sources>` files to your configuration
directory:

1. ``data-cds-era5.yml``

    Enable access to ERA5 data stored in the
    `Copernicus Climate Data Store (CDS) <https://cds.climate.copernicus.eu/>`__:

    .. code-block:: console

        esmvaltool config copy data-cds-era5.yml

    Then edit the copied file and insert your
    `CDS API key <https://cds.climate.copernicus.eu/how-to-api>`__.

2. ``data-gcs-era5.yml``

    Enable access to `ERA5 data stored in Google Cloud Storage
    <https://github.com/google-research/arco-era5>`__:

    .. code-block:: console

        esmvaltool config copy data-gcs-era5.yml

    The file content is:

    .. literalinclude:: ../configurations/data-gcs-era5.yml
        :language: yaml
        :caption: Contents of ``data-gcs-era5.yml``

    Install `gcsfs <https://pypi.org/project/gcsfs/>`_ alongside ESMValCore to
    use this data source.

"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import xarray as xr

from esmvalcore.dataset import _ismatch
from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    import iris.cube

    from esmvalcore.typing import Facets, FacetValue


@dataclass
class XarrayDataset(DataElement):
    """A dataset that can load data using :meth:`xarray <xarray.open_dataset>`."""

    name: str
    """A unique name identifying the data."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    store: str = field(repr=False)
    """The data store containing the data."""

    options: dict[str, Any] = field(repr=False, default_factory=dict)
    """Options to use when opening the data."""

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
        options = self.options | {
            # Do not decode the time coordininate because Iris needs the
            # encoded values.
            "decode_times": False,
        }
        dataset = xr.open_dataset(self.store, **options)
        dataset.attrs["source_file"] = str(self.store)
        # Cache the attributes.
        self.attributes = copy.deepcopy(dataset.attrs)
        return dataset_to_iris(dataset)


@dataclass
class XarrayDataSource(DataSource):
    """Data source that can find data using :meth:`xarray <xarray.open_dataset>`."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    variables: dict[str, str]
    """Mapping between the ESMValCore short_names and dataset variable names."""

    facets: Facets
    """Facets that will be added to the data elements found using this data source."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    store: str
    """The store used to load the data."""

    options: dict[str, Any] = field(default_factory=dict)
    """Options to use when opening the data store."""

    def find_data(self, **facets: FacetValue) -> list[XarrayDataset]:
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
        normalized_facets = {
            facet: [str(values)]
            if isinstance(values, str | int | float)
            else values
            for facet, values in facets.items()
            if facet in ["short_name", *self.facets]
        }
        if not all(
            any(_ismatch(self.facets[k], v) for v in values)
            for k, values in normalized_facets.items()
            if k != "short_name"
        ):
            return []

        # Would it be useful to use `xr.open_datatree` instead?
        # https://tutorial.xarray.dev/intermediate/storage_formats.html#how-does-zarr-relate-to-xarray

        search_options = self.options | {
            # Avoid loading the coordinate data.
            "create_default_indexes": False,
            # Computing chunk sizes is expensive for datasets with many variables.
            "chunks": None,
        }
        search_dataset = xr.open_dataset(self.store, **search_options)
        available_variables = {
            self.variables.get(v, v): v for v in search_dataset.data_vars
        }
        short_names = normalized_facets.get("short_name", ["*"])
        matching_variables = [
            v
            for v in available_variables
            for s in short_names
            if _ismatch(v, s)
        ]
        if not matching_variables:
            self.debug_info = (
                f"No variables matching {', '.join(sorted(short_names))} found "
                f"in {self.store}, available variables are: "
                f"{', '.join(sorted(available_variables))}."
            )
        result = []
        for short_name in matching_variables:
            dataset_facets = copy.deepcopy(self.facets)
            dataset_facets["short_name"] = short_name
            data_var = available_variables[short_name]
            dataset = XarrayDataset(
                name=f"{self.store}/{short_name}",
                store=self.store,
                options={
                    "drop_variables": [
                        v
                        for v in available_variables.values()
                        if v != data_var
                    ],
                    **self.options,
                },
                facets=dataset_facets,
            )
            result.append(dataset)
        return result
