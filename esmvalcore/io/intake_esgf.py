from dataclasses import dataclass, field
from numbers import Number

import intake_esgf.projects
import iris.cube
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris
from esmvalcore.typing import Facets, FacetValue


@dataclass
class IntakeESGFDataset(DataElement):
    name: str
    """A unique name identifying the data."""

    facets: Facets
    """Facets are key-value pairs describing the data."""

    catalog: ESGFCatalog

    def __hash__(self) -> int:
        return hash(self.name)

    def prepare(self) -> None:
        """Prepare the data for access."""
        self.catalog.to_path_dict()

    def to_iris(self, ignore_warnings=None) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        iris.cube.CubeList
            The loaded data.
        """
        dataset = self.catalog.to_dataset_dict(
            minimal_keys=False,
            add_measures=False,
            quiet=True,
        )[self.name]
        return dataset_to_iris(dataset, ignore_warnings=ignore_warnings)


@dataclass
class IntakeESGFDataSource(DataSource):
    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    facets: dict[str, str | list[str]]

    values: dict[str, dict[str, str]] = field(default_factory=dict)

    debug_info: str = ""
    """A string containing debug information when no data is found."""

    catalog: ESGFCatalog = field(default_factory=ESGFCatalog)

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
        :obj:`list` of :obj:`esmvalcore.io.intake_esgf.IntakeESGFDataset`
            A list of data elements that have been found.
        """
        # Normalize facets so all values are `list[str]`.
        our_facets = {
            facet: [str(values)]
            if isinstance(values, str | Number | bool)
            else values
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
        # TODO: filter by timerange
        try:
            self.catalog.search(**esgf_facets, quiet=True)
        except NoSearchResults:
            self.debug_info = ", ".join(
                [
                    f"{k}={v if isinstance(v, list) else [v]}"
                    for k, v in self.catalog.last_search.items()
                ],
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
            dataset_facets = {
                our_facet: [
                    inverse_values.get(our_facet, {}).get(v, v)
                    for v in row[their_facet]
                ]
                for our_facet, their_facet in self.facets.items()
                if their_facet in row
            }
            dataset_facets = {
                f: v[0] if len(v) == 1 else v
                for f, v in dataset_facets.items()
            }
            dataset = IntakeESGFDataset(
                name=dataset_id,
                facets=dataset_facets,  # type: ignore[arg-type]
                catalog=cat,
            )
            result.append(dataset)
        return result
