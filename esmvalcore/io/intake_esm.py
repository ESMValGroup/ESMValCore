"""Find and load data from `intake-esm <https://intake-esm.readthedocs.io/>`_ catalogs.

Configure a catalog URL or local path as a data source using, for example,
``esmvaltool config copy data-intake-esm-gcs.yml``. The catalog is opened when
the data source is first searched.
"""

from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake

from esmvalcore.io.local import _parse_period, _truncate_dates
from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    import iris.cube
    from intake_esm.core import esm_datastore

    from esmvalcore.typing import Facets, FacetValue


__all__ = [
    "IntakeEsmDataSource",
    "IntakeEsmDataset",
]


@dataclass
class IntakeEsmDataset(DataElement):
    """A catalog selection that can be loaded as Iris cubes."""

    name: str
    """A unique name identifying the data."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    catalog: esm_datastore = field(repr=False)
    """The intake-esm catalog describing this data."""

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

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        : The loaded data.
        """
        if len(self.catalog) != 1:
            msg = (
                f"Expected one intake-esm dataset for '{self.name}', "
                f"found {len(self.catalog)}. Map the missing identity facets "
                "or correct the catalog aggregation rules."
            )
            raise ValueError(msg)
        path_column = self.catalog.esmcat.assets.column_name
        files = self.catalog.df[path_column].dropna().unique().tolist()

        dataset = self.catalog.to_dask(**self.to_dask_kwargs)
        # A catalog can add dimensions that are not part of the CMOR variable.
        # Only discard dimensions explicitly named by its configuration.
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
        return dataset_to_iris(dataset, filepath=source_files)


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

    catalog_filters: dict[str, str | int | float | list[str | int | float]] = (
        field(default_factory=dict)
    )
    """Fixed catalog selections applied before recipe facets."""

    to_dask_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    """Keyword arguments passed to ``intake_esm.esm_datastore.to_dask``."""

    squeeze_dimensions: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    """Catalog aggregation dimensions to drop when they have length one."""

    time_separator: str = field(repr=False, default="-")
    """Separator used in catalog asset time ranges, e.g. ``185002-185501``."""

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
        self.debug_info = ""
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
        debug_query = copy.deepcopy(query)

        catalog = self._get_catalog()
        _validate_catalog_columns(catalog, self.facets, self.catalog_filters)

        # A catalog time_range describes asset coverage, not an exact recipe
        # value. Select overlapping assets after the other catalog filters.
        time_column = self.facets.get("timerange")
        requested_times = query.pop(time_column, None) if time_column else None

        # Resolve both exact values and globs against catalog metadata. This
        # preserves native catalog types (for example, numeric versions), and
        # avoids intake-esm interpreting a glob as a regular expression.
        fixed_query = {
            column: [str(value) for value in values]
            if isinstance(values, list | tuple)
            else [str(values)]
            for column, values in self.catalog_filters.items()
        }
        fixed_query = _resolve_catalog_query(catalog, fixed_query)
        res = catalog.search(**fixed_query) if fixed_query else catalog
        query = _resolve_catalog_query(res, query)
        res = res.search(**query) if query else res
        if requested_times and res.df.shape[0]:
            if any(
                any(char in value for char in "*?[")
                for value in requested_times
            ):
                selected_ranges = None
            else:
                selected_ranges = [
                    value
                    for value in res.df[time_column].dropna().unique()
                    if any(
                        _ranges_overlap(
                            requested,
                            str(value),
                            self.time_separator,
                        )
                        for requested in requested_times
                    )
                ]
            if selected_ranges is not None:
                res = res.search(**{time_column: selected_ranges})

        if not len(res):
            self.debug_info = (
                "`intake_esm.esm_datastore().search("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, list) else f"{k}='{v}'"
                        for k, v in debug_query.items()
                    ],
                )
                + ")` did not return any results."
            )
            return []

        # Return one element per logical dataset, regardless of how the
        # publisher chose the intake-esm aggregation keys.
        result: list[IntakeEsmDataset] = []

        inverse_values = {
            our_facet: {
                their_value: our_value
                for our_value, their_value in self.values[our_facet].items()
            }
            for our_facet in self.values
        }

        identity_columns = list(
            dict.fromkeys(
                column
                for facet, column in self.facets.items()
                if facet != "timerange"
            ),
        )
        groups = (
            res.df.groupby(identity_columns, dropna=False, sort=False)
            if identity_columns
            else [((), res.df)]
        )
        for _, rows in groups:
            selection = {
                column: [rows[column].iloc[0]] for column in identity_columns
            }
            cat = res.search(**selection) if selection else res
            if len(cat.df) != len(rows):
                msg = (
                    f"Catalog '{catalog.esmcat.id}' returned ambiguous rows for "
                    f"{selection}. Check the facet mapping and catalog values."
                )
                raise ValueError(msg)
            if len(cat) != 1:
                msg = (
                    f"Catalog '{catalog.esmcat.id}' groups one logical dataset "
                    f"into {len(cat)} intake-esm keys. Map the distinguishing "
                    "facet or correct its aggregation rules."
                )
                raise ValueError(msg)

            dataset_facets = {
                our_facet: inverse_values.get(our_facet, {}).get(
                    rows[column].iloc[0],
                    rows[column].iloc[0],
                )
                for our_facet, column in self.facets.items()
                if our_facet != "timerange"
            }
            identity = ",".join(
                f"{facet}={value}"
                for facet, value in sorted(dataset_facets.items())
                if facet != "version"
            )
            result.append(
                IntakeEsmDataset(
                    name=f"{self.project}:{identity}",
                    facets=dataset_facets,
                    catalog=cat,
                    to_dask_kwargs=copy.deepcopy(self.to_dask_kwargs),
                    squeeze_dimensions=tuple(self.squeeze_dimensions),
                ),
            )
        return result


def _validate_catalog_columns(
    catalog: esm_datastore,
    facets: dict[str, str],
    filters: dict[str, Any],
) -> None:
    """Check that configured catalog columns can be searched safely."""
    missing_facets = set(facets.values()) - set(catalog.df.columns)
    if missing_facets:
        msg = (
            f"intake-esm catalog '{catalog.esmcat.id}' is missing "
            "configured facet columns: "
            f"{', '.join(sorted(missing_facets))}"
        )
        raise ValueError(msg)
    missing_filters = set(filters) - set(catalog.df.columns)
    if missing_filters:
        msg = (
            f"intake-esm catalog '{catalog.esmcat.id}' is missing "
            "configured filter columns: "
            f"{', '.join(sorted(missing_filters))}"
        )
        raise ValueError(msg)
    iterable_columns = (
        set(facets.values()) & catalog.esmcat.columns_with_iterables
    )
    if iterable_columns:
        msg = (
            f"intake-esm catalog '{catalog.esmcat.id}' has list-valued "
            f"mapped facet columns: {', '.join(sorted(iterable_columns))}. "
            "The adapter requires one value per row for mapped facets."
        )
        raise ValueError(msg)


def _resolve_catalog_query(
    catalog: esm_datastore,
    query: dict[str, list[str]],
) -> dict[str, list[Any]]:
    """Resolve string selections to native catalog values, including globs."""
    return {
        facet: [
            match
            for value in values
            for match in catalog.df[facet].dropna().unique()
            if (
                fnmatch.fnmatchcase(str(match), value)
                if any(char in value for char in "*?[")
                else str(match) == value
            )
        ]
        for facet, values in query.items()
    }


def _ranges_overlap(requested: str, asset: str, separator: str) -> bool:
    """Check overlap using the same mixed-resolution rules as local files."""
    request_start, request_end = _parse_period(requested)
    asset_dates = asset.split(separator)
    if len(asset_dates) != 2:
        msg = (
            f"Catalog time range {asset!r} must contain two dates "
            f"separated by {separator!r}."
        )
        raise ValueError(msg)
    asset_start, asset_end = asset_dates
    request_start_int, asset_end_int = _truncate_dates(
        request_start,
        asset_end,
    )
    request_end_int, asset_start_int = _truncate_dates(
        request_end,
        asset_start,
    )
    return (
        asset_start_int <= request_end_int
        and asset_end_int >= request_start_int
    )
