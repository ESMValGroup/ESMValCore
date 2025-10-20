"""Access data using `intake-esgf <https://intake-esgf.readthedocs.io>`_.

The default :ref:`configuration <config_overview>` is

.. literalinclude:: ../configurations/intake-esgf-data.yml
   :language: yaml

"""

import copy
from dataclasses import dataclass, field
from typing import Any

import intake_esgf
import intake_esgf.exceptions
import iris.cube
import isodate

from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris
from esmvalcore.local import _parse_period
from esmvalcore.typing import Facets, FacetValue

__all__ = [
    "IntakeESGFDataSource",
    "IntakeESGFDataset",
]


@dataclass
class IntakeESGFDataset(DataElement):
    """A dataset that can be used to load data found using intake-esgf_."""

    name: str
    """A unique name identifying the data."""

    facets: Facets
    """Facets are key-value pairs that were used to find this data."""

    catalog: intake_esgf.ESGFCatalog
    """The intake-esgf catalog describing this data."""

    _attributes: dict[str, Any] | None = field(init=False, default=None)

    def __hash__(self) -> int:
        """Return a number uniquely representing the data element."""
        return hash(self.name)

    def prepare(self) -> None:
        """Prepare the data for access."""
        self.catalog.to_path_dict()

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

    def to_iris(self, ignore_warnings=None) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        :
            The loaded data.
        """
        files = self.catalog.to_path_dict(
            minimal_keys=False,
            quiet=True,
        )[self.name]
        dataset = self.catalog.to_dataset_dict(
            minimal_keys=False,
            add_measures=False,
            quiet=True,
        )[self.name]
        # Store the local paths in the attributes for easier debugging.
        dataset.attrs["source_file"] = ", ".join(str(f) for f in files)
        # Cache the attributes.
        self.attributes = copy.deepcopy(dataset.attrs)
        return dataset_to_iris(dataset, ignore_warnings=ignore_warnings)


@dataclass
class IntakeESGFDataSource(DataSource):
    """Data source that can be used to find data using intake-esgf."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    facets: dict[str, str]
    """Mapping between the ESMValCore and ESGF facet names."""

    values: dict[str, dict[str, str]] = field(default_factory=dict)
    """Mapping between the ESMValCore and ESGF facet values."""

    debug_info: str = field(init=False, default="")
    """A string containing debug information when no data is found."""

    catalog: intake_esgf.ESGFCatalog = field(
        init=False,
        default_factory=intake_esgf.ESGFCatalog,
    )
    """The intake-esgf catalog used to find data."""

    def __post_init__(self):
        self.catalog.project = intake_esgf.projects.projects[
            self.project.lower()
        ]

    def find_data(self, **facets: FacetValue) -> list[IntakeESGFDataset]:
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
        # Normalize facets so all values are `list[str]`.
        our_facets = {
            facet: [str(values)] if isinstance(values, str | int) else values
            for facet, values in facets.items()
        }
        # Translate "our" facets to ESGF facets and "our" values to ESGF values.
        esgf_facets = {
            their_facet: [
                self.values.get(our_facet, {}).get(v, v)
                for v in our_facets[our_facet]
            ]
            for our_facet, their_facet in self.facets.items()
            if our_facet in our_facets
        }
        if (
            "timerange" in facets and "*" not in facets["timerange"]  # type: ignore[operator]
        ):
            start, end = _parse_period(facets["timerange"])
            esgf_facets["file_start"] = isodate.date_isoformat(
                isodate.parse_date(start.split("T")[0]),
            )
            esgf_facets["file_end"] = isodate.date_isoformat(
                isodate.parse_date(end.split("T")[0]),
            )
        # Search ESGF.
        try:
            self.catalog.search(**esgf_facets, quiet=True)
        except intake_esgf.exceptions.NoSearchResults:
            self.debug_info = (
                "intake_esgf.ESGFCatalog.search("
                + ", ".join(
                    [
                        f"{k}={v if isinstance(v, list) else [v]}"
                        for k, v in self.catalog.last_search.items()
                    ],
                )
                + ") did not return any results."
            )
            return []

        # Return a list of datasets, with one IntakeESGFDataset per dataset_id.
        result: list[IntakeESGFDataset] = []

        # These are the keys in the dict[str, xarray.Dataset] returned by
        # `intake_esgf.ESGFCatalog.to_dataset_dict`. Taken from:
        # https://github.com/esgf2-us/intake-esgf/blob/c34124e54078e70ef271709a6d158edb22bcdb96/intake_esgf/catalog.py#L523-L528
        self.catalog.df["key"] = self.catalog.df.apply(
            lambda row: ".".join(
                [row[f] for f in self.catalog.project.master_id_facets()],
            ),
            axis=1,
        )
        inverse_values = {
            our_facet: {
                their_value: our_value
                for our_value, their_value in self.values[our_facet].items()
            }
            for our_facet in self.values
        }
        for _, row in self.catalog.df.iterrows():
            dataset_id = row["key"]
            # Subset the catalog to a single dataset.
            cat = self.catalog.clone()
            cat.file_start = self.catalog.file_start
            cat.file_end = self.catalog.file_end
            cat.df = self.catalog.df[self.catalog.df.key == dataset_id]
            # Discard all but the latest version. It is not clear how/if
            # `intake_esgf.ESGFCatalog.to_dataset_dict` supports multiple versions.
            cat.df = cat.df[cat.df.version == cat.df.version.max()]
            cat.project = self.catalog.project
            if "short_name" in our_facets:
                cat.last_search[self.facets["short_name"]] = [
                    self.values.get("short_name", {}).get(v, v)
                    for v in our_facets["short_name"]
                ]
            # Retrieve "our" facets associated with the dataset_id.
            dataset_facets = {}
            for our_facet, esgf_facet in self.facets.items():
                if esgf_facet in row:
                    esgf_values = row[esgf_facet]
                    if isinstance(esgf_values, str):
                        esgf_values = [esgf_values]
                    our_values = [
                        inverse_values.get(our_facet, {}).get(v, v)
                        for v in esgf_values
                    ]
                    if len(our_values) == 1:
                        our_values = our_values[0]
                    dataset_facets[our_facet] = our_values
            dataset = IntakeESGFDataset(
                name=dataset_id,
                facets=dataset_facets,  # type: ignore[arg-type]
                catalog=cat,
            )
            result.append(dataset)
        return result
