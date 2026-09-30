"""Find and load data from `intake-esm <https://intake-esm.readthedocs.io/>`_ catalogs.

Configure a catalog URL or local path as a data source using, for example,
``esmvaltool config copy data-intake-esm-gcs.yml``. The catalog is opened when
the data source is first searched.
"""

from __future__ import annotations

import copy
import fnmatch
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake
import numpy as np

from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    import iris.cube
    from intake_esm.core import esm_datastore
    from intake_esm.source import ESMDataSource

    from esmvalcore.typing import Facets, FacetValue


__all__ = [
    "IntakeEsmDataSource",
    "IntakeEsmDataset",
]


def _to_path_dict(
    esm_datastore: esm_datastore,
    quiet: bool = False,
) -> dict[str, list[str | Path]]:
    """Return the current search as a dictionary of paths to files.

    This method does not exist on intake-ESM's esm_datastore, so we implement it here.
    """
    if not esm_datastore.keys() and not quiet:
        warnings.warn(
            "There are no datasets to load! Returning an empty dictionary.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    def _to_pathlist(source: ESMDataSource) -> list[str | Path]:
        return source.df[source.path_column_name].to_list()

    return {key: _to_pathlist(val) for key, val in esm_datastore.items()}


@dataclass
class IntakeEsmDataset(DataElement):
    """A catalog selection that can be loaded as Iris cubes."""

    name: str
    """A unique name identifying the data."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    catalog: esm_datastore = field(repr=False)
    """The intake-esm catalog describing this data."""

    catalog_key: str | None = field(default=None, repr=False)
    """The intake-esm key, which can differ from the data element name."""

    to_dask_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    """Keyword arguments passed to ``intake_esm.esm_datastore.to_dask``."""

    squeeze_dimensions: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    """Catalog aggregation dimensions to drop when they have length one."""

    _attributes: dict[str, Any] | None = field(
        init=False,
        repr=False,
        default=None,
    )

    def __hash__(self) -> int:
        """Return a number uniquely representing the data element."""
        return hash((self.name, self.facets.get("version")))

    def prepare(self) -> None:
        """Prepare the data for access.

        For intake-esm, no preparation is needed.
        """

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

    def to_iris(self, quiet: bool = True) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Arguments
        ---------
        quiet : bool, optional
            If True, suppress warnings when no datasets are found. Default is True.

        Returns
        -------
        : The loaded data.
        """
        files = _to_path_dict(self.catalog, quiet=quiet)[
            self.catalog_key or self.name
        ]

        dataset = self.catalog.to_dask(**self.to_dask_kwargs)
        for dimension in self.squeeze_dimensions:
            if dimension in dataset.sizes:
                if dataset.sizes[dimension] != 1:
                    msg = (
                        f"Cannot squeeze dimension '{dimension}' with "
                        f"length {dataset.sizes[dimension]} in '{self.name}'."
                    )
                    raise ValueError(msg)
                dataset = dataset.squeeze(dim=dimension, drop=True)
        # Preserve the asset URLs or paths for debugging and CMOR fixes.
        source_files = ", ".join(str(file) for file in files)
        dataset.attrs["source_file"] = source_files
        # Cache the attributes.
        self.attributes = copy.deepcopy(dataset.attrs)
        cubes = dataset_to_iris(dataset, filepath=source_files)
        # ncdata can turn scalar Zarr attributes into zero-dimensional arrays.
        # CMOR metadata fixes expect scalar values for attributes such as
        # branch_time_in_child and branch_time_in_parent.
        for cube in cubes:
            for name in ("branch_time_in_child", "branch_time_in_parent"):
                value = cube.attributes.get(name)
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    cube.attributes[name] = value.item()
        return cubes


@dataclass
class IntakeEsmDataSource(DataSource):
    """Data source that finds data in an intake-esm datastore."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    facets: dict[str, str]
    """Mapping between the ESMValCore and intake-esm facet names."""

    catalog: str | Path | esm_datastore = field(
        repr=False,
    )
    """An intake-esm catalog, or a URL/path to one."""

    to_dask_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    """Keyword arguments passed to ``intake_esm.esm_datastore.to_dask``."""

    squeeze_dimensions: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    """Catalog aggregation dimensions to drop when they have length one."""

    time_separator: str = field(repr=False, default="/")
    """Separator used in catalog time ranges, e.g. ``185002-185501``."""

    values: dict[str, dict[str, str]] = field(default_factory=dict)
    """Mapping between the ESMValCore and intake-esm facet values."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    def _get_catalog(self) -> esm_datastore:
        """Open a configured catalog once, when it is first needed."""
        if isinstance(self.catalog, str | Path):
            self.catalog = intake.open_esm_datastore(str(self.catalog))
        return self.catalog

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
            facet: [str(values)]
            if isinstance(values, str | int | float)
            else [str(value) for value in values]
            for facet, values in facets.items()
            if facet in self.facets and values is not None
        }

        # A lone '*' does not constrain the search.
        normalized_facets = {
            facet: values
            for facet, values in normalized_facets.items()
            if "*" not in values
        }

        # Translate ESMValCore facet names and values to catalog values.
        query = {
            their_facet: [
                self.values.get(our_facet, {}).get(v, v)
                for v in normalized_facets[our_facet]
            ]
            for our_facet, their_facet in self.facets.items()
            if our_facet in normalized_facets
        }

        if self.time_separator != "/" and self.facets.get("timerange"):
            time_facet = self.facets["timerange"]
            if time_facet in query:
                query[time_facet] = [
                    value.replace("/", self.time_separator)
                    for value in query[time_facet]
                ]

        catalog = self._get_catalog()

        # Match globs against catalog metadata before searching. Passing a glob
        # directly to intake-esm makes it interpret '*' as a regular expression.
        query = {
            facet: [
                match
                for value in values
                for match in (
                    [
                        candidate
                        for candidate in catalog.df[facet].dropna().unique()
                        if fnmatch.fnmatchcase(str(candidate), value)
                    ]
                    if any(char in value for char in "*?[")
                    else [value]
                )
            ]
            for facet, values in query.items()
        }

        res = catalog.search(**query) if query else catalog

        if not len(res):
            self.debug_info = (
                "`intake_esm.esm_datastore().search("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, list) else f"{k}='{v}'"
                        for k, v in query.items()
                    ],
                )
                + ")` did not return any results."
            )
            return []

        # Return one data element per set of mapped facets. Intake-esm keys can
        # combine multiple versions or ensemble members into one dataset.
        result: list[IntakeEsmDataset] = []

        inverse_values = {
            our_facet: {
                their_value: our_value
                for our_value, their_value in self.values[our_facet].items()
            }
            for our_facet in self.values
        }

        for key in sorted(res):
            esm_datasource = res[key]
            path_col = esm_datasource.path_column_name
            df = esm_datasource.df
            varying_columns = [
                column
                for column in self.facets.values()
                if column in df and df[column].nunique(dropna=False) > 1
            ]
            groups = (
                df.groupby(varying_columns, dropna=False, sort=True)
                if varying_columns
                else [((), df)]
            )
            for _, rows in groups:
                selection = {
                    column: [rows[column].iloc[0]]
                    for column in varying_columns
                }
                selection[path_col] = rows[path_col].unique().tolist()
                cat = res.search(**selection)

                dataset_facets = {}
                for our_facet, esm_facet in self.facets.items():
                    if esm_facet in cat.df:
                        esm_values = cat.df[esm_facet].unique().tolist()
                        our_values = [
                            inverse_values.get(our_facet, {}).get(value, value)
                            for value in esm_values
                        ]
                        dataset_facets[our_facet] = our_values[0]

                identity = ",".join(
                    f"{facet}={value}"
                    for facet, value in sorted(dataset_facets.items())
                    if facet != "version"
                )
                result.append(
                    IntakeEsmDataset(
                        name=f"{key}:{identity}",
                        facets=dataset_facets,
                        catalog=cat,
                        catalog_key=key,
                        to_dask_kwargs=copy.deepcopy(self.to_dask_kwargs),
                        squeeze_dimensions=tuple(self.squeeze_dimensions),
                    ),
                )
        return result
