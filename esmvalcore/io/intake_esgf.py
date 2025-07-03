from dataclasses import dataclass, field

import intake_esgf.projects
import iris.cube
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris
from esmvalcore.typing import FacetValue


@dataclass
class IntakeESGFDataset(DataElement):
    name: str
    """A unique name identifying the data."""

    facets: dict[str, FacetValue]
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

    facets: dict[str, str]

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
        # Translate "our" facets to ESGF facets
        esgf_facets = {
            self.values.get(k, {}).get(v, v): facets[k]
            for k, v in self.facets.items()
            if k in facets and facets[k] != "*"
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

        self.catalog.df["key"] = self.catalog.df.apply(
            lambda row: ".".join(
                [row[f] for f in self.catalog.project.master_id_facets()],
            ),
            axis=1,
        )
        inverse_values = {
            facet: {v: k}
            for facet in self.values
            for k, v in self.values[facet].items()
        }
        datasets = []
        for _, row in self.catalog.df.iterrows():
            dataset_id = row["key"]
            # Subset the catalog to a single dataset.
            cat = self.catalog.clone()
            cat.project = self.catalog.project
            cat.df = self.catalog.df[self.catalog.df.key == dataset_id]
            facets = {
                k: inverse_values.get(k, {}).get(row[v], row[v])
                for k, v in self.facets.items()
            }
            dataset = IntakeESGFDataset(
                name=dataset_id,
                facets=facets,
                catalog=cat,
            )
            datasets.append(dataset)
        return datasets
