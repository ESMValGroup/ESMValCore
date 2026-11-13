"""Access data using :meth:`xarray.open_dataset`.

To use this module, copy one of the following
:ref:`configuration <config-data-sources>` files to your configuration
directory:

1. ``data-cds-era5.yml``

    Enable access to ERA5 data stored in the
    `Copernicus Climate Data Store (CDS) <https://cds.climate.copernicus.eu/>`__:

    .. code-block:: console

        esmvaltool config copy data-cds-era5.yml

    The file content is:

    .. literalinclude:: ../configurations/data-cds-era5.yml
        :language: yaml
        :caption: Contents of ``data-cds-era5.yml``

    Then set the environment variable ``CDSAPI_KEY`` to your personal
    `CDS API key <https://cds.climate.copernicus.eu/how-to-api>`__.

2. ``data-gcs-era5.yml``

    Enable access to `ERA5 data stored in Google Cloud Storage
    <https://github.com/google-research/arco-era5>`__:

    .. code-block:: console

        esmvaltool config copy data-gcs-era5.yml

3. ``data-edh-cmip6.yml`` and ``data-edh-era5.yml``

    Enable access to `CMIP6 <https://earthdatahub.destine.eu/collections/cmip6>`__
    and `monthly ERA5
    <https://earthdatahub.destine.eu/collections/era5/datasets/era5-single-levels-atmosphere-monthly>`__
    data stored in the `Earth Data Hub <https://earthdatahub.destine.eu>`__ of
    Destination Earth:

    .. code-block:: console

        esmvaltool config copy data-edh-cmip6.yml
        esmvaltool config copy data-edh-era5.yml

    The Earth Data Hub requires a DestinE account. Add your personal access
    token to your ``~/.netrc`` file as described in the configuration files.

Finding data
------------

The stores that a data source can read from are configured with one of the
following options:

- ``store``: a single store.
- ``stores``: a list of stores.
- ``index``: the URL of a JSON file that contains a list of stores, relative to
  the directory containing the index file.
- ``url_template``: a template for the store URLs, e.g.
  ``"s3://bucket/{dataset}/{exp}.zarr"``. If none of the options above is
  given, the stores are found by replacing the facets in the template by their
  values, or by ``*`` for facets that are not specified, and listing the
  matching stores with :func:`fsspec.open_files`. If one of the options above
  is given as well, the template is only used to read facets from the URLs of
  the stores.

The ``variables`` option maps an ESMValCore ``short_name`` to a variable in the
store, or to a list of variables in the store that are needed to load it.
Variables in the store that are not mentioned in ``variables`` can be loaded
using their name in the store as ``short_name``.

Facets that do not depend on the store or variable can be configured with the
``facets`` option. If the store has a global attribute that lists the datasets
it contains, the facets can be read from it with the
``facets_from_attribute`` option, e.g.

.. code-block:: yaml

    facets_from_attribute:
      attribute: dataset_ids
      template: "{project}.{activity}.{institute}.{dataset}.{exp}.{ensemble}.{mip}.{short_name}.{grid}.{version}"

If a store contains several datasets along a dimension, a facet can be used to
select one of them with the ``select`` option, which maps a facet to the name
of the dimension, e.g. ``select: {exp: experiment_id}``.

The ``name`` of each data element is the store URL followed by the
``short_name`` and the values of the facets in ``select``, unless a
``name_template`` is configured, e.g.
``"{project}.{activity}.{institute}.{dataset}.{exp}.{ensemble}.{mip}.{short_name}.{grid}"``.
Data elements with the same name and ``version`` facet that are found by
different data sources are considered to be the same data.

String values in ``options`` can contain environment variables, written as
``${NAME}``, e.g. to read an access token from the environment.

The ``options`` are passed to :func:`xarray.open_dataset`. When data is
loaded, only the variables that are needed are opened, and by default it is
split into chunks of a reasonable size with ``chunks: auto``. Set ``chunks``
in ``options`` to use other chunks, e.g. ``chunks: {}`` to use the chunks of
the store.
"""

from __future__ import annotations

import copy
import datetime
import itertools
import json
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cftime
import fsspec
import numpy as np
import xarray as xr

from esmvalcore.dataset import _isglob, _ismatch
from esmvalcore.io.local import (
    _get_caps_options,
    _parse_period,
    _replace_tag,
    _select_files,
)
from esmvalcore.io.protocol import DataElement, DataSource
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator

    import iris.cube

    from esmvalcore.typing import Facets, FacetValue

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(r"{([^}]*)}")

# CF attributes that contain a space separated list of variable names.
_LIST_ATTRIBUTES = (
    "ancillary_variables",
    "bounds",
    "climatology",
    "coordinates",
)
# CF attributes that contain a list of "key: variable" pairs.
_KEY_VALUE_ATTRIBUTES = ("cell_measures", "formula_terms", "grid_mapping")


def _expand_env_vars(value: Any) -> Any:  # noqa: ANN401
    """Replace ``${NAME}`` in strings by the value of environment variables."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _template_tags(template: str) -> set[str]:
    """Return the facets used in a template."""
    return {
        _get_caps_options(tag)[0] for tag in _TAG_PATTERN.findall(template)
    }


def _format_template(
    template: str,
    facets: Facets,
    *,
    missing: str | None = None,
) -> list[str]:
    """Replace the facets in ``template`` by their values.

    Facets with a list of values result in one string per value. Facets that
    are not in ``facets`` are replaced by ``missing``, or raise a
    :obj:`KeyError` if ``missing`` is ``None``.
    """
    results = [template]
    for original_tag in set(_TAG_PATTERN.findall(template)):
        tag = _get_caps_options(original_tag)[0]
        if tag in facets:
            value = facets[tag]
        elif missing is not None:
            value = missing
        else:
            msg = (
                f"Unable to complete template '{template}' because the facet "
                f"'{tag}' has not been specified."
            )
            raise KeyError(msg)
        results = _replace_tag(results, original_tag, value)
    return sorted(results)


def _template_to_regex(template: str) -> re.Pattern[str]:
    """Convert a template to a regular expression that extracts the facets.

    Tags are matched by any characters except ``/``. A tag that appears more
    than once must have the same value everywhere. The wildcard ``*`` matches
    any characters except ``/``.
    """
    pattern = ""
    used: set[str] = set()
    for i, part in enumerate(_TAG_PATTERN.split(template)):
        if i % 2 == 0:
            pattern += re.escape(part).replace(r"\*", "[^/]*")
            continue
        tag = _get_caps_options(part)[0]
        if tag in used:
            pattern += f"(?P={tag})"
        else:
            pattern += f"(?P<{tag}>[^/]+?)"
            used.add(tag)
    return re.compile(pattern)


def _parse_template(
    regex: re.Pattern[str],
    text: str,
) -> dict[str, str] | None:
    """Extract facets from ``text``, or return ``None`` if it does not match."""
    if (match := regex.fullmatch(text)) is None:
        return None
    return match.groupdict()


def _matches(facets: Facets, requested: dict[str, list[str]]) -> bool:
    """Check that ``facets`` matches the requested facets.

    Only facets that are in ``facets`` are compared.
    """
    return all(
        any(_ismatch(str(facets[k]), v) for v in values)
        for k, values in requested.items()
        if k in facets and k != "timerange"
    )


def _referenced_names(variable: xr.DataArray) -> Iterator[str]:
    """Yield the names of the variables that ``variable`` refers to."""
    attrs = variable.attrs | variable.encoding
    for attribute in _LIST_ATTRIBUTES:
        yield from str(attrs.get(attribute, "")).split()
    for attribute in _KEY_VALUE_ATTRIBUTES:
        words = str(attrs.get(attribute, "")).split()
        if len(words) == 1:
            # grid_mapping may be a single variable name.
            yield words[0]
        else:
            yield from (word for word in words if not word.endswith(":"))


def _required_names(
    dataset: xr.Dataset,
    names: Iterable[str],
) -> set[Hashable]:
    """Return ``names`` and every variable they refer to, recursively."""
    required: set[Hashable] = set()
    todo: list[Hashable] = list(names)
    while todo:
        name = todo.pop()
        if name in required or name not in dataset.variables:
            continue
        required.add(name)
        variable = dataset.variables[name]
        todo.extend(dataset[name].coords if name in dataset else [])
        todo.extend(_referenced_names(dataset[name]))
        todo.extend(
            str(dim) for dim in variable.dims if dim in dataset.variables
        )
    return required


def _time_coordinate(dataset: xr.Dataset) -> str | None:
    """Return the name of the time coordinate of an undecoded dataset."""
    for name, variable in dataset.variables.items():
        units = str(variable.attrs.get("units", ""))
        if (
            variable.ndim == 1
            and variable.dims[0] == name
            and " since " in units
        ):
            return str(name)
    return None


def _num2date(variable: xr.Variable, values: Any) -> Any:  # noqa: ANN401
    """Convert the values of an undecoded time coordinate to dates."""
    return cftime.num2date(
        values,
        units=variable.attrs["units"],
        calendar=variable.attrs.get("calendar", "standard"),
    )


def _date2num(variable: xr.Variable, date: cftime.datetime) -> float:
    """Convert a date to the encoding of an undecoded time coordinate."""
    return float(
        cftime.date2num(
            date,
            units=variable.attrs["units"],
            calendar=variable.attrs.get("calendar", "standard"),
        ),
    )


def _parse_date(date: str) -> list[int]:
    """Split a date like ``YYYYMMDDTHHMMSS``, or a part of it, into numbers."""
    digits = re.sub("[^0-9]", "", date)
    widths = (4, 2, 2, 2, 2, 2)
    values = []
    pos = 0
    for width in widths:
        if pos >= len(digits):
            break
        values.append(int(digits[pos : pos + width]))
        pos += width
    return values


def _date_bounds(
    timerange: str,
    calendar: str,
) -> tuple[cftime.datetime, cftime.datetime]:
    """Return the start and the (exclusive) end date of ``timerange``.

    The dates have the resolution given in ``timerange``, like ESMValCore uses
    when it selects files, e.g. ``2000/2001`` selects all of 2000 and 2001.
    The exact selection is done later by the ``clip_timerange`` preprocessor
    step.
    """
    start, end = (_parse_date(date) for date in _parse_period(timerange))
    defaults = [1, 1, 1, 0, 0, 0]
    start_date = cftime.datetime(
        *start,
        *defaults[len(start) :],
        calendar=calendar,
    )
    end_date = cftime.datetime(*end, *defaults[len(end) :], calendar=calendar)
    if len(end) == 1:
        end_date = end_date.replace(year=end_date.year + 1)
    elif len(end) == 2:
        month = end_date.month % 12 + 1
        year = end_date.year + (end_date.month == 12)
        end_date = end_date.replace(year=year, month=month)
    else:
        units = ("days", "hours", "minutes", "seconds")
        end_date += datetime.timedelta(**{units[len(end) - 3]: 1})
    return start_date, end_date


def _select_timerange(dataset: xr.Dataset, timerange: str) -> xr.Dataset:
    """Select the time points in ``timerange`` from an undecoded dataset."""
    if "*" in timerange or (time := _time_coordinate(dataset)) is None:
        return dataset
    variable = dataset.variables[time]
    calendar = variable.attrs.get("calendar", "standard")
    start, end = _date_bounds(timerange, calendar)
    values = variable.values
    selection = (values >= _date2num(variable, start)) & (
        values < _date2num(variable, end)
    )
    return dataset.isel({time: np.flatnonzero(selection)})


def _get_timerange(dataset: xr.Dataset) -> str | None:
    """Return the timerange of an undecoded dataset."""
    if (time := _time_coordinate(dataset)) is None:
        return None
    variable = dataset.variables[time]
    if variable.size == 0:
        return None
    dates = _num2date(variable, variable[[0, -1]].values)
    start, end = (d.strftime("%Y%m%dT%H%M%S") for d in dates)
    return f"{start}/{end}"


@dataclass
class XarrayDataset(DataElement):
    """A dataset that can load data using :meth:`xarray <xarray.open_dataset>`."""

    name: str
    """A unique name identifying the dataset."""

    facets: Facets = field(repr=False)
    """Facets are key-value pairs that were used to find this data."""

    store: str = field(repr=False)
    """The data store containing the data."""

    variables: list[str] = field(repr=False, default_factory=list)
    """The variables to load from the store."""

    drop_variables: list[str] = field(repr=False, default_factory=list)
    """Variables in the store that are not needed to load ``variables``.

    These are not opened, which is much faster for stores with many
    variables.
    """

    select: dict[str, Any] = field(repr=False, default_factory=dict)
    """Values to select along dimensions of the store."""

    timerange: str | None = field(repr=False, default=None)
    """The requested time range."""

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

    def _open(self) -> xr.Dataset:
        """Open the store and select the requested data."""
        options = (
            # Use dask chunks of a reasonable size by default.
            {"chunks": "auto"}
            | self.options
            | {
                # Do not decode the time coordininate because Iris needs the
                # encoded values.
                "decode_times": False,
                "drop_variables": [
                    *self.options.get("drop_variables", []),
                    *self.drop_variables,
                ],
            }
        )
        dataset = xr.open_dataset(self.store, **options)
        selection = {
            dim: value
            for dim, value in self.select.items()
            if dim in dataset.dims
        }
        if selection:
            dataset = dataset.sel(selection).drop_vars(
                list(selection),
                errors="ignore",
            )
        if self.variables:
            required = _required_names(dataset, self.variables)
            dataset = dataset.drop_vars(
                [name for name in dataset.data_vars if name not in required],
            )
        if self.timerange is not None:
            dataset = _select_timerange(dataset, self.timerange)
        return dataset

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        :
            The loaded data.
        """
        dataset = self._open()
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

    variables: dict[str, str | list[str]] = field(default_factory=dict)
    """Mapping from ESMValCore ``short_name`` to the variable(s) in the store."""

    facets: Facets = field(default_factory=dict)
    """Facets that will be added to the data elements found using this data source."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    store: str | None = None
    """A single store used to load the data."""

    stores: list[str] = field(default_factory=list)
    """A list of stores used to load the data."""

    index: str | None = None
    """URL of a JSON file containing a list of stores."""

    url_template: str | None = None
    """Template used to find stores and to read facets from their URLs."""

    facets_from_attribute: dict[str, str] | None = None
    """Read facets from a global attribute of the store using a template."""

    select: dict[str, str] = field(default_factory=dict)
    """Mapping from facet to a dimension in the store to select along."""

    name_template: str | None = None
    """Template for the name of the data elements."""

    options: dict[str, Any] = field(default_factory=dict)
    """Options to use when opening the data store."""

    _metadata: dict[str, xr.Dataset] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        """Check the configuration."""
        sources = [
            name
            for name, value in [
                ("store", self.store),
                ("stores", self.stores),
                ("index", self.index),
            ]
            if value
        ]
        if len(sources) > 1:
            msg = (
                f"Data source '{self.name}' can only have one of 'store', "
                f"'stores', or 'index', got {', '.join(sources)}."
            )
            raise ValueError(msg)
        if not sources and self.url_template is None:
            msg = (
                f"Data source '{self.name}' needs one of 'store', 'stores', "
                f"'index', or 'url_template'."
            )
            raise ValueError(msg)
        if self.facets_from_attribute is not None and not {
            "attribute",
            "template",
        }.issubset(self.facets_from_attribute):
            msg = (
                f"'facets_from_attribute' of data source '{self.name}' needs "
                f"the keys 'attribute' and 'template'."
            )
            raise ValueError(msg)
        self.options = _expand_env_vars(self.options)

    @property
    def _storage_options(self) -> dict[str, Any]:
        return self.options.get("storage_options", {})

    def _candidate_urls(self, facets: Facets) -> list[str]:
        """Return the URLs of the stores that may contain the data."""
        if self.store:
            return [self.store]
        if self.stores:
            return list(self.stores)
        if self.index:
            with fsspec.open(self.index, **self._storage_options) as file:
                entries = json.load(file)
            base = posixpath.dirname(self.index)
            return [
                entry if "://" in entry else posixpath.join(base, entry)
                for entry in entries
            ]
        if self.url_template is None:  # pragma: no cover
            return []
        urls = []
        for pattern in _format_template(
            self.url_template,
            facets,
            missing="*",
        ):
            if not _isglob(pattern):
                urls.append(pattern)
                continue
            fs, path = fsspec.core.url_to_fs(pattern, **self._storage_options)
            # Keep the protocol only if the template has one, so the URLs
            # still match the template.
            urls.extend(
                fs.unstrip_protocol(p) if "://" in pattern else p
                for p in sorted(fs.glob(path))
            )
        return urls

    def _open_metadata(self, url: str) -> xr.Dataset:
        """Open a store without loading data, using a cache."""
        if url not in self._metadata:
            options = self.options | {
                # Avoid loading the coordinate data.
                "create_default_indexes": False,
                # Computing chunk sizes is expensive for datasets with many
                # variables.
                "chunks": None,
                "decode_times": False,
            }
            self._metadata[url] = xr.open_dataset(url, **options)
        return self._metadata[url]

    def _store_variables(self, short_name: str) -> list[str]:
        """Return the variables in the store needed to load ``short_name``."""
        value = self.variables.get(short_name, short_name)
        return [value] if isinstance(value, str) else list(value)

    def _available_short_names(self, dataset: xr.Dataset) -> list[str]:
        """Return the short names that can be loaded from ``dataset``."""
        mapped = {
            name
            for short_name in self.variables
            for name in self._store_variables(short_name)
        }
        short_names = [
            short_name
            for short_name in self.variables
            if all(
                name in dataset.data_vars
                for name in self._store_variables(short_name)
            )
        ]
        # Bounds are part of the variable they belong to.
        bounds = {
            name
            for variable in dataset.variables.values()
            for attribute in ("bounds", "climatology")
            for name in str(variable.attrs.get(attribute, "")).split()
        }
        short_names.extend(
            str(name)
            for name in dataset.data_vars
            if name not in mapped and name not in bounds
        )
        return short_names

    def _facets_from_attribute(self, dataset: xr.Dataset) -> list[Facets]:
        """Read facets from a global attribute of ``dataset``."""
        if self.facets_from_attribute is None:
            return []
        attribute = self.facets_from_attribute["attribute"]
        value = dataset.attrs.get(attribute, [])
        entries = (
            value.replace(",", " ").split()
            if isinstance(value, str)
            else value
        )
        regex = _template_to_regex(self.facets_from_attribute["template"])
        result: list[Facets] = []
        for entry in entries:
            if (parsed := _parse_template(regex, str(entry))) is None:
                logger.debug(
                    "Unable to read facets from '%s' in attribute '%s'",
                    entry,
                    attribute,
                )
            else:
                result.append(parsed)  # type: ignore[arg-type]
        return result

    def _datasets_in_store(
        self,
        url: str,
        url_facets: Facets,
        requested: dict[str, list[str]],
    ) -> list[Facets]:
        """Return the facets of each dataset in the store."""
        dataset = self._open_metadata(url)
        base_facets = copy.deepcopy(self.facets) | url_facets
        available = self._available_short_names(dataset)

        if self.facets_from_attribute is not None:
            candidates = [
                base_facets | facets
                for facets in self._facets_from_attribute(dataset)
                if facets.get("short_name") in available
            ]
        else:
            selections: list[list[tuple[str, str]]] = [
                [
                    (facet, str(value))
                    for value in np.atleast_1d(dataset[dim].values)
                ]
                for facet, dim in self.select.items()
                if dim in dataset.dims
            ]
            candidates = [
                base_facets | {"short_name": short_name} | dict(combination)
                for short_name in available
                for combination in itertools.product(*selections)
            ]
        candidates = [c for c in candidates if _matches(c, requested)]
        if candidates and (timerange := _get_timerange(dataset)):
            for candidate in candidates:
                candidate["timerange"] = timerange
        return candidates

    def _data_element_name(self, url: str, facets: Facets) -> str:
        """Return the name of a data element."""
        if self.name_template is not None:
            return _format_template(self.name_template, facets)[0]
        return "/".join(
            [
                url,
                str(facets["short_name"]),
                *(
                    str(facets[facet])
                    for facet in self.select
                    if facet in facets
                ),
            ],
        )

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
        requested = {
            facet: [str(values)]
            if isinstance(values, str | int | float)
            else [str(v) for v in values]  # type: ignore[union-attr]
            for facet, values in facets.items()
        }
        if not _matches(self.facets, requested):
            return []

        url_regex = (
            None
            if self.url_template is None
            else _template_to_regex(self.url_template)
        )
        url_facets_requested: Facets = {
            k: v[0]
            for k, v in requested.items()
            if len(v) == 1 and not _isglob(v[0])
        }

        result: list[XarrayDataset] = []
        searched: list[str] = []
        for url in self._candidate_urls(url_facets_requested):
            url_facets: Facets = {}
            if url_regex is not None:
                parsed = _parse_template(url_regex, url)
                if parsed is None:
                    logger.debug("Skipping %s: does not match template", url)
                    continue
                url_facets = parsed  # type: ignore[assignment]
                if not _matches(url_facets, requested):
                    continue
            searched.append(url)
            for dataset_facets in self._datasets_in_store(
                url,
                url_facets,
                requested,
            ):
                short_name = str(dataset_facets["short_name"])
                variables = self._store_variables(short_name)
                metadata = self._open_metadata(url)
                required = _required_names(metadata, variables)
                result.append(
                    XarrayDataset(
                        name=self._data_element_name(url, dataset_facets),
                        facets=dataset_facets,
                        store=url,
                        variables=variables,
                        drop_variables=[
                            str(name)
                            for name in metadata.variables
                            if name not in required
                        ],
                        select={
                            dim: dataset_facets[facet]
                            for facet, dim in self.select.items()
                            if facet in dataset_facets
                        },
                        timerange=str(facets["timerange"])
                        if "timerange" in facets
                        else None,
                        options=self.options,
                    ),
                )

        if "timerange" in facets:
            result = _select_files(result, facets["timerange"])

        if not result:
            self.debug_info = (
                f"No data matching {facets} found in "
                + (", ".join(searched) if searched else "any store")
                + "."
            )
        return result
