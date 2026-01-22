"""Access data using `intake-esm <https://intake-esgf.readthedocs.io>`_.

.. note::

    It is highly recommended that you take a moment to
    :doc:`configure intake-esm <intake_esm:configure>` before using it
    with ESMValCore. Make sure to set ``local_cache`` to a path where
    it can store downloaded files and if (some) Esm data is already
    available on your system, point ``esg_dataroot`` to it. If you are
    missing certain search results, you may want to choose a different
    index node for searching the Esm.

Run the command ``esmvaltool config copy data-intake-esm.yml`` to update
your :ref:`configuration <config-data-sources>` to use this module. This will
create a file with the following content in your configuration directory:

.. literalinclude:: ../configurations/data-intake-esm.yml
   :language: yaml
   :caption: Contents of ``data-intake-esm.yml``

"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake_esm

# import intake_esm.exceptions
import isodate

from esmvalcore.dataset import _isglob, _ismatch
from esmvalcore.io.local import _parse_period
from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    import iris.cube

    from esmvalcore.typing import Facets, FacetValue


__all__ = [
    "IntakeEsmDataSource",
    "IntakeEsmDataset",
]


class _CachingCatalog(intake_esm.core.esm_datastore):
    """An Intake-ESM catalog that caches to_path_dict results."""

    def __init__(self):
        super().__init__()
        self._result = {}

    @classmethod
    def from_catalog(
        cls,
        catalog: intake_esm.core.esm_datastore,
    ) -> _CachingCatalog:
        """Create a CachingCatalog from an existing EsmCatalog."""
        cat = cls()
        cat.indices = catalog.indices
        cat.local_cache = catalog.local_cache
        cat.esg_dataroot = catalog.esg_dataroot
        cat.file_start = catalog.file_start
        cat.file_end = catalog.file_end
        cat.project = catalog.project
        cat.df = catalog.df
        return cat

    def to_path_dict(
        self,
        prefer_streaming: bool = False,
        globus_endpoint: str | None = None,
        globus_path: Path = Path("/"),
        minimal_keys: bool = True,
        ignore_facets: None | str | list[str] = None,
        separator: str = ".",
        quiet: bool = False,
    ) -> dict[str, list[str | Path]]:
        """Return the current search as a dictionary of paths to files."""
        kwargs = {
            "prefer_streaming": prefer_streaming,
            "globus_endpoint": globus_endpoint,
            "globus_path": globus_path,
            "minimal_keys": minimal_keys,
            "ignore_facets": ignore_facets,
            "separator": separator,
            "quiet": quiet,
        }
        key = tuple((k, v) for k, v in kwargs.items() if k != "quiet")
        if key not in self._result:
            self._result[key] = super().to_path_dict(**kwargs)
        return self._result[key]


@dataclass
class IntakeEsmDataset(DataElement):
    """A dataset that can be used to load data found using intake-esm_."""

    name: str
    """A unique name identifying the data."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    catalog: intake_esm.core.esm_datastore = field(repr=False)
    """The intake-esm catalog describing this data."""

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
        self.catalog.to_path_dict(minimal_keys=False)
        for index in self.catalog.indices:
            # Set the sessions to None to avoid issues with pickling
            # requests_cache.CachedSession objects when max_parallel_tasks > 1.
            # After the prepare step, the sessions for interacting with the
            # search indices are not needed anymore as all file paths required
            # to load the data have been found. To make sure we do not
            # accidentally use the sessions later on, we set them to None
            # instead of e.g. requests.Session objects.
            #
            # This seems the safest/fastest solution as it avoids accessing the
            # sqlite database backing the cached_requests.CachedSession from
            # multiple processes on multiple machines.
            index.session = None

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
        return dataset_to_iris(dataset)


@dataclass
class IntakeEsmDataSource(DataSource):
    """Data source that can be used to find data using intake-esm."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    facets: dict[str, str]
    """Mapping between the ESMValCore and Esm facet names."""

    values: dict[str, dict[str, str]] = field(default_factory=dict)
    """Mapping between the ESMValCore and Esm facet values."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    catalog: intake_esm.core.esm_datastore = field(
        init=False,
        repr=False,
        default_factory=intake_esm.core.esm_datastore,
    )
    """The intake-esm catalog used to find data."""

    def __post_init__(self):
        self.catalog.project = intake_esm.projects.projects[self.project.lower()]

    def find_data(self, **facets: FacetValue) -> list[IntakeEsmDataset]:
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
        # Select searchable facets and normalize so all values are `list[str]`.
        normalized_facets = {
            facet: [str(values)] if isinstance(values, str | int | float) else values
            for facet, values in facets.items()
            if facet in self.facets
        }
        # Filter out glob patterns as these are not supported by intake-esm.
        non_glob_facets = {
            facet: values
            for facet, values in normalized_facets.items()
            if not any(_isglob(v) for v in values)
        }
        # Translate "our" facets to Esm facets and "our" values to Esm values.
        query = {
            their_facet: [
                self.values.get(our_facet, {}).get(v, v)
                for v in non_glob_facets[our_facet]
            ]
            for our_facet, their_facet in self.facets.items()
            if our_facet in non_glob_facets
        }
        if (
            "timerange" in facets and not _isglob(facets["timerange"])  # type: ignore[operator]
        ):
            start, end = _parse_period(facets["timerange"])
            query["file_start"] = isodate.date_isoformat(
                isodate.parse_date(start.split("T")[0]),
            )
            query["file_end"] = isodate.date_isoformat(
                isodate.parse_date(end.split("T")[0]),
            )
        # Search Esm.
        try:
            self.catalog.search(**query, quiet=True)
        except intake_esm.exceptions.NoSearchResults:
            self.debug_info = (
                "`intake_esm.core.esm_datastore().search("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, list) else f"{k}='{v}'"
                        for k, v in query.items()
                    ],
                )
                + ")` did not return any results."
            )
            return []

        # Return a list of datasets, with one IntakeEsmDataset per dataset_id.
        result: list[IntakeEsmDataset] = []

        # These are the keys in the dict[str, xarray.Dataset] returned by
        # `intake_esm.core.esm_datastore.to_dataset_dict`. Taken from:
        # https://github.com/esgf2-us/intake-esm/blob/c34124e54078e70ef271709a6d158edb22bcdb96/intake_esm/catalog.py#L523-L528
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
            # Use a caching catalog to avoid searching the indices after
            # calling the EsmFile.prepare method.
            cat = _CachingCatalog.from_catalog(self.catalog)
            # Subset the catalog to a single dataset.
            cat.df = cat.df[cat.df.key == dataset_id]
            # Ensure only the requested variable is included in the dataset.
            # https://github.com/esgf2-us/intake-esm/blob/18437bff5ee75acaaceef63093101223b4692259/intake_esm/catalog.py#L544-L552
            if "short_name" in normalized_facets:
                cat.last_search[self.facets["short_name"]] = [
                    self.values.get("short_name", {}).get(v, v)
                    for v in normalized_facets["short_name"]
                ]
            # Retrieve "our" facets associated with the dataset_id.
            dataset_facets = {"version": [f"v{row['version']}"]}
            for our_facet, esgf_facet in self.facets.items():
                if esgf_facet in row:
                    esgf_values = row[esgf_facet]
                    if isinstance(esgf_values, str):
                        esgf_values = [esgf_values]
                    our_values = [
                        inverse_values.get(our_facet, {}).get(v, v) for v in esgf_values
                    ]
                    dataset_facets[our_facet] = our_values
            # Only return datasets that match the glob patterns.
            if all(
                any(
                    _ismatch(v, p)
                    for v in dataset_facets[f]
                    for p in normalized_facets[f]
                )
                for f in dataset_facets
                if f in normalized_facets
            ):
                dataset = IntakeEsmDataset(
                    name=dataset_id,
                    facets={
                        k: v[0] if len(v) == 1 else v for k, v in dataset_facets.items()
                    },  # type: ignore[arg-type]
                    catalog=cat,
                )
                result.append(dataset)
        return result
