"""Find files on the local filesystem.

Example configuration to find CMIP6 data on a personal computer:

.. literalinclude:: ../configurations/data-local.yml
    :language: yaml
    :caption: Contents of ``data-local.yml``
    :start-at: projects:
    :end-before: CMIP5:

The module will find files matching the :func:`glob.glob` pattern formed by
``rootpath/dirname_template/filename_template``, where the facets defined
inside the curly braces of the templates are replaced by their values
from the :class:`~esmvalcore.dataset.Dataset` or the :ref:`recipe <recipe>`
plus any facet-value pairs that can be automatically added using
:meth:`~esmvalcore.dataset.Dataset.augment_facets`.
Note that the name of the data source, ``local-data`` in the example above,
must be unique within each project but can otherwise be chosen freely.

To start using this module on a personal computer, copy the example
configuration file into your configuration directory by running the command:

.. code-block:: bash

    esmvaltool config copy data-local.yml

and tailor it for your own system if needed.

Example configuration files for popular HPC systems and some
:ref:`supported climate models <read_native_models>` are also available. View
the list of available files by running the command:

.. code-block:: bash

    esmvaltool config list

Further information is available in :ref:`config-data-sources`.

"""

from __future__ import annotations

import copy
import itertools
import logging
import os
import os.path
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

import iris.cube
import iris.fileformats.cf
import isodate
from cf_units import Unit
from netCDF4 import Dataset

import esmvalcore.io.protocol
from esmvalcore.exceptions import RecipeError
from esmvalcore.iris_helpers import ignore_warnings_context

if TYPE_CHECKING:
    from collections.abc import Iterable

    from netCDF4 import Variable

    from esmvalcore.typing import Facets, FacetValue

logger = logging.getLogger(__name__)


def _get_from_pattern(
    pattern: str,
    date_range_pattern: str,
    stem: str,
    group: str,
) -> tuple[str | None, str | None]:
    """Get time, date or datetime from date range patterns in file names."""
    # Next string allows to test that there is an allowed delimiter (or
    # string start or end) close to date range (or to single date)
    start_point: str | None = None
    end_point: str | None = None
    context = r"(?:^|[-_]|$)"

    # First check for a block of two potential dates
    date_range_pattern_with_context = context + date_range_pattern + context
    daterange = re.search(date_range_pattern_with_context, stem)
    if not daterange:
        # Retry with extended context for CMIP3
        context = r"(?:^|[-_.]|$)"
        date_range_pattern_with_context = (
            context + date_range_pattern + context
        )
        daterange = re.search(date_range_pattern_with_context, stem)

    if daterange:
        start_point = daterange.group(group)
        end_group = f"{group}_end"
        end_point = daterange.group(end_group)
    else:
        # Check for single dates in the filename
        single_date_pattern = context + pattern + context
        dates = re.findall(single_date_pattern, stem)
        if len(dates) == 1:
            start_point = end_point = dates[0][0]
        elif len(dates) > 1:
            # Check for dates at start or (exclusive or) end of filename
            start = re.search(r"^" + pattern, stem)
            end = re.search(pattern + r"$", stem)
            if start and not end:
                start_point = end_point = start.group(group)
            elif end:
                start_point = end_point = end.group(group)

    return start_point, end_point


def _get_var_name(variable: Variable) -> str:
    """Get variable name (following Iris' Cube.name())."""
    for attr in ("standard_name", "long_name"):
        if attr in variable.ncattrs():
            return str(variable.getncattr(attr))
    return str(variable.name)


def _get_start_end_date_from_filename(
    file: str | Path,
) -> tuple[str | None, str | None]:
    """Get the start and end dates as a string from a file name.

    Examples of allowed dates: 1980, 198001, 1980-01, 19801231, 1980-12-31,
    1980123123, 19801231T23, 19801231T2359, 19801231T235959, 19801231T235959Z
    (ISO 8601).

    Dates must be surrounded by '-', '_' or '.' (the latter is used by CMIP3
    data), or string start or string end (after removing filename suffix).

    Look first for two dates separated by '-', '_' or '_cat_' (the latter is
    used by CMIP3 data), then for one single date, and if there are multiple,
    for one date at start or end.

    Parameters
    ----------
    file:
        The file to read the start and end data from.

    Returns
    -------
    tuple[str, str]
        The start and end date.

    Raises
    ------
    ValueError
        Start or end date cannot be determined.
    """
    start_date = end_date = None

    # Build regex
    time_pattern = (
        r"(?P<hour>[0-2][0-9]"
        r"(?P<minute>[0-5][0-9]"
        r"(?P<second>[0-5][0-9])?)?Z?)"
    )
    date_pattern = (
        r"(?P<year>[0-9]{4})"
        r"(?P<month>-?[01][0-9]"
        r"(?P<day>-?[0-3][0-9]"
        rf"(T?{time_pattern})?)?)?"
    )
    datetime_pattern = rf"(?P<datetime>{date_pattern})"
    end_datetime_pattern = datetime_pattern.replace(">", "_end>")

    # Dates can either be delimited by '-', '_', or '_cat_' (the latter for
    # CMIP3)
    date_range_pattern = (
        datetime_pattern + r"[-_](?:cat_)?" + end_datetime_pattern
    )

    # Find dates using the regex
    start_date, end_date = _get_from_pattern(
        datetime_pattern,
        date_range_pattern,
        Path(file).stem,
        "datetime",
    )
    return start_date, end_date


def _get_start_end_date(file: str | Path) -> tuple[str, str]:
    """Get the start and end dates as a string from a file.

    This function first tries to read the dates from the filename and only
    if that fails, it will try to read them from the content of the file.

    Parameters
    ----------
    file:
        The file to read the start and end data from.

    Returns
    -------
    tuple[str, str]
        The start and end date.

    Raises
    ------
    ValueError
        Start or end date cannot be determined.
    """
    start_date, end_date = _get_start_end_date_from_filename(file)

    # As final resort, try to get the dates from the file contents
    if (
        (start_date is None or end_date is None)
        and isinstance(file, (str, Path))
        and Path(file).exists()
    ):
        logger.debug("Must load file %s for daterange ", file)
        with Dataset(file) as dataset:
            for variable in dataset.variables.values():
                var_name = _get_var_name(variable)
                attrs = variable.ncattrs()
                if (
                    var_name == "time"
                    and "units" in attrs
                    and "calendar" in attrs
                ):
                    time_units = Unit(
                        variable.getncattr("units"),
                        calendar=variable.getncattr("calendar"),
                    )
                    start_date = isodate.date_isoformat(
                        time_units.num2date(variable[0]),
                        format=isodate.isostrf.DATE_BAS_COMPLETE,
                    )
                    end_date = isodate.date_isoformat(
                        time_units.num2date(variable[-1]),
                        format=isodate.isostrf.DATE_BAS_COMPLETE,
                    )
                    break

    if start_date is None or end_date is None:
        msg = (
            f"File {file} datetimes do not match a recognized pattern and "
            f"time coordinate can not be read from the file"
        )
        raise ValueError(msg)

    # Remove potential '-' characters from datetimes
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    return start_date, end_date


def _dates_to_timerange(start_date: int | str, end_date: int | str) -> str:
    """Convert ``start_date`` and ``end_date`` to ``timerange``.

    Note
    ----
    This function ensures that dates in years format follow the pattern YYYY
    (i.e., that they have at least 4 digits). Other formats, such as wildcards
    (``'*'``) and relative time ranges (e.g., ``'P6Y'``) are used unchanged.

    Parameters
    ----------
    start_date:
        Start date.
    end_date:
        End date.

    Returns
    -------
    str
        ``timerange`` in the form ``'start_date/end_date'``.
    """
    start_date = str(start_date)
    end_date = str(end_date)

    # Pad years with 0s if not wildcard or relative time range
    if start_date != "*" and not start_date.startswith("P"):
        start_date = start_date.zfill(4)
    if end_date != "*" and not end_date.startswith("P"):
        end_date = end_date.zfill(4)

    return f"{start_date}/{end_date}"


def _replace_years_with_timerange(variable: dict[str, Any]) -> None:
    """Set `timerange` tag from tags `start_year` and `end_year`."""
    start_year = variable.get("start_year")
    end_year = variable.get("end_year")
    if start_year and end_year:
        variable["timerange"] = _dates_to_timerange(start_year, end_year)
    elif start_year:
        variable["timerange"] = _dates_to_timerange(start_year, start_year)
    elif end_year:
        variable["timerange"] = _dates_to_timerange(end_year, end_year)
    variable.pop("start_year", None)
    variable.pop("end_year", None)


def _parse_period(timerange: FacetValue) -> tuple[str, str]:
    """Parse `timerange` values given as duration periods.

    Sum the duration periods to the `timerange` value given as a
    reference point in order to compute the start and end dates needed
    for file selection.
    """
    if not isinstance(timerange, str):
        msg = f"`timerange` should be a `str`, got {type(timerange)}"
        raise TypeError(msg)
    start_date: str | None = None
    end_date: str | None = None
    time_format = None
    datetime_format = (
        isodate.DATE_BAS_COMPLETE + "T" + isodate.TIME_BAS_COMPLETE
    )
    if timerange.split("/")[0].startswith("P"):
        try:
            end_date = isodate.parse_datetime(timerange.split("/")[1])
            time_format = datetime_format
        except isodate.ISO8601Error:
            end_date = isodate.parse_date(timerange.split("/")[1])
            time_format = isodate.DATE_BAS_COMPLETE
        delta = isodate.parse_duration(timerange.split("/")[0])
        start_date = end_date - delta
    elif timerange.split("/")[1].startswith("P"):
        try:
            start_date = isodate.parse_datetime(timerange.split("/")[0])
            time_format = datetime_format
        except isodate.ISO8601Error:
            start_date = isodate.parse_date(timerange.split("/")[0])
            time_format = isodate.DATE_BAS_COMPLETE
        delta = isodate.parse_duration(timerange.split("/")[1])
        end_date = start_date + delta

    if time_format == datetime_format:
        start_date = str(
            isodate.datetime_isoformat(start_date, format=datetime_format),
        )
        end_date = str(
            isodate.datetime_isoformat(end_date, format=datetime_format),
        )
    elif time_format == isodate.DATE_BAS_COMPLETE:
        start_date = str(
            isodate.date_isoformat(start_date, format=time_format),
        )
        end_date = str(isodate.date_isoformat(end_date, format=time_format))

    if start_date is None:
        start_date = timerange.split("/")[0]
    if end_date is None:
        end_date = timerange.split("/")[1]

    return start_date, end_date


def _truncate_dates(date: str, file_date: str) -> tuple[int, int]:
    """Truncate dates of different lengths and convert to integers.

    This allows to compare the dates chronologically. For example, this allows
    comparisons between the formats 'YYYY' and 'YYYYMM', and 'YYYYMM' and
    'YYYYMMDD'.

    Warning
    -------
    This function assumes that the years in ``date`` and ``file_date`` have the
    same number of digits. If this is not the case, pad the dates with leading
    zeros (e.g., use ``date='0100'`` and ``file_date='199901'`` for a correct
    comparison).
    """
    date = re.sub("[^0-9]", "", date)
    file_date = re.sub("[^0-9]", "", file_date)
    if len(date) < len(file_date):
        file_date = file_date[0 : len(date)]
    elif len(date) > len(file_date):
        date = date[0 : len(file_date)]

    return int(date), int(file_date)


def _select_files(
    filenames: Iterable[LocalFile],
    timerange: FacetValue,
) -> list[LocalFile]:
    """Select files containing data between a given timerange.

    If the timerange is given as a period, the file selection occurs
    taking only the years into account.

    Otherwise, the file selection occurs taking into account the time
    resolution of the file.
    """
    if not isinstance(timerange, str):
        msg = f"`timerange` should be a `str`, got {type(timerange)}"
        raise TypeError(msg)
    if "*" in timerange:
        # TODO: support * combined with a period
        return list(filenames)

    selection: list[LocalFile] = []

    for filename in filenames:
        if "timerange" not in filename.facets:
            # Gracefully handle files where no timerange could be determined.
            selection.append(filename)
            continue

        start_date, end_date = _parse_period(timerange)
        start, end = filename.facets["timerange"].split("/")  # type: ignore[union-attr]

        start_date_int, end_int = _truncate_dates(start_date, end)
        end_date_int, start_int = _truncate_dates(end_date, start)
        if start_int <= end_date_int and end_int >= start_date_int:
            selection.append(filename)

    return selection


def _replace_tags(
    paths: str | list[str],
    variable: Facets,
) -> list[Path]:
    """Replace tags in the config-developer's file with actual values."""
    pathset: Iterable[str]
    if isinstance(paths, str):
        pathset = {paths.strip("/")}
    else:
        pathset = {path.strip("/") for path in paths}
    tlist: set[str] = set()
    for path in pathset:
        tlist = tlist.union(re.findall(r"{([^}]*)}", path))
    if "sub_experiment" in variable:
        new_paths: set[str] = set()
        for path in pathset:
            new_paths.update(
                (
                    re.sub(r"(\b{ensemble}\b)", r"{sub_experiment}-\1", path),
                    re.sub(r"({ensemble})", r"{sub_experiment}-\1", path),
                ),
            )
            tlist.add("sub_experiment")
        pathset = new_paths

    for original_tag in tlist:
        tag, _, _ = _get_caps_options(original_tag)

        if tag in variable:
            replacewith = variable[tag]
        elif tag == "version":
            replacewith = "*"
        else:
            msg = (
                f"Dataset key '{tag}' must be specified for {variable}, check "
                f"your recipe entry and/or extra facet file(s)"
            )
            raise RecipeError(msg)
        pathset = _replace_tag(pathset, original_tag, replacewith)
    return [Path(p) for p in pathset]


def _replace_tag(
    paths: Iterable[str],
    tag: str,
    replacewith: FacetValue,
) -> list[str]:
    """Replace tag by replacewith in paths."""
    _, lower, upper = _get_caps_options(tag)
    result: list[str] = []
    if isinstance(replacewith, (list, tuple)):
        for item in replacewith:
            result.extend(_replace_tag(paths, tag, item))
    else:
        text = _apply_caps(str(replacewith), lower, upper)
        result.extend(p.replace("{" + tag + "}", text) for p in paths)
    return list(set(result))


def _get_caps_options(tag: str) -> tuple[str, bool, bool]:
    lower = False
    upper = False
    if tag.endswith(".lower"):
        lower = True
        tag = tag[0:-6]
    elif tag.endswith(".upper"):
        upper = True
        tag = tag[0:-6]
    return tag, lower, upper


def _apply_caps(original: str, lower: bool, upper: bool) -> str:
    if lower:
        return original.lower()
    if upper:
        return original.upper()
    return original


@dataclass(order=True)
class LocalDataSource(esmvalcore.io.protocol.DataSource):
    """Data source for finding files on a local filesystem."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    rootpath: Path
    """The path where the directories are located."""

    dirname_template: str
    """The template for the directory names."""

    filename_template: str
    """The template for the file names."""

    ignore_warnings: list[dict[str, Any]] | None = field(default_factory=list)
    """Warnings to ignore when loading the data.

    The list should contain :class:`dict`s with keyword arguments that
    will be passed to the :func:`warnings.filterwarnings` function when
    calling :meth:`LocalFile.to_iris`.
    """

    def __post_init__(self) -> None:
        """Set further attributes."""
        self.rootpath = Path(os.path.expandvars(self.rootpath)).expanduser()
        self._regex_pattern = self._templates_to_regex()

    def _get_glob_patterns(self, **facets: FacetValue) -> list[Path]:
        """Compose the globs that will be used to look for files."""
        dirname_globs = _replace_tags(self.dirname_template, facets)
        filename_globs = _replace_tags(self.filename_template, facets)
        return sorted(
            self.rootpath / d / f
            for d in dirname_globs
            for f in filename_globs
        )

    def find_data(self, **facets: FacetValue) -> list[LocalFile]:
        """Find data locally.

        Parameters
        ----------
        **facets :
            Find data matching these facets.

        Returns
        -------
        :
            A list of files.

        """
        facets = dict(facets)
        if "original_short_name" in facets:
            facets["short_name"] = facets["original_short_name"]

        globs = self._get_glob_patterns(**facets)
        self.debug_info = "No files found matching glob pattern " + "\n".join(
            str(g) for g in globs
        )
        logger.debug("Looking for files matching %s", globs)

        files: list[LocalFile] = []
        for glob_ in globs:
            for filename in glob(str(glob_)):
                file = LocalFile(filename)
                file.facets.update(
                    self._path2facets(
                        file,
                        add_timerange=facets.get("frequency", "fx") != "fx",
                    ),
                )
                file.ignore_warnings = self.ignore_warnings
                files.append(file)

        files = _filter_versions_called_latest(files)

        if "version" not in facets:
            files = _select_latest_version(files)

        files.sort()  # sorting makes it easier to see what was found

        if "timerange" in facets:
            found_files = bool(files)
            files = _select_files(files, facets["timerange"])
            if not files and found_files:
                self.debug_info += (
                    f" within the requested timerange {facets['timerange']}"
                )

        return files

    def _path2facets(self, path: Path, add_timerange: bool) -> dict[str, str]:
        """Extract facets from path."""
        facets: dict[str, str] = {}

        if (match := re.search(self._regex_pattern, str(path))) is not None:
            for facet, value in match.groupdict().items():
                if value:
                    facets[facet] = value

        if add_timerange:
            try:
                start_date, end_date = _get_start_end_date(path)
            except ValueError:
                pass
            else:
                facets["timerange"] = _dates_to_timerange(start_date, end_date)

        return facets

    def _templates_to_regex(self) -> str:
        r"""Convert template strings to regex pattern.

        The resulting regex pattern can be used to extract facets from paths
        using :func:`re.search`.

        Note
        ----
        Facets must not contain "/" or "_".

        Examples
        --------
        - rootpath: "/root"
          dirname_template: "{f2.upper}"
          filename_template: "{f3}[._]{f4}*"
          --> regex_pattern:
          "/root/(?P<f2>[^_/]*?)/(?P<f3>[^_/]*?)[\._](?P<f4>[^_/]*?).*?"
        - rootpath: "/root"
          dirname_template: "{f1}/{f1}-{f2}"
          filename_template: "*.nc"
          --> regex_pattern:
          "/root/(?P<f1>[^_/]*?)/(?P=f1)\-(?P<f2>[^_/]*?)/.*?\.nc"
        - rootpath: "/root"
          dirname_template: "{f1}/{f2}{f3}"
          filename_template: "*.nc"
          --> regex_pattern:
          "/root/(?P<f1>[^_/]*?)/(?:[^_/]*?)/.*?\.nc"

        """
        dirname_template = self.dirname_template
        filename_template = self.filename_template

        # Templates must not be absolute paths (i.e., start with /), otherwise
        # the roopath is ignored (see
        # https://docs.python.org/3/library/pathlib.html#operators)
        if self.dirname_template.startswith(os.sep):
            dirname_template = dirname_template[1:]
        if self.filename_template.startswith(os.sep):
            filename_template = filename_template[1:]

        pattern = re.escape(
            str(self.rootpath / dirname_template / filename_template),
        )

        # Remove all tags that are in between other tags, e.g.,
        # {tag1}{tag2}{tag3} -> {tag1}{tag3} (there is no way to reliably
        # extract facets from those)
        pattern = re.sub(r"(?<=\})\\\{[^\}]+?\\\}(?=\\(?=\{))", "", pattern)

        # Replace consecutive tags, e.g. {tag1}{tag2} with non-capturing groups
        # (?:[^_/]*?) (there is no way to reliably extract facets from those)
        # Note: This assumes that facets do NOT contain / or _
        pattern = re.sub(
            r"\\\{[^\{]+?\}\\\{[^\}]+?\\\}",
            rf"(?:[^_{os.sep}]*?)",
            pattern,
        )

        # Convert tags {tag} to named capture groups (?P<tag>[^_/]*?); for
        # duplicates use named backreferences (?P=tag)
        # Note: This assumes that facets do NOT contain / or _
        already_used_tags: set[str] = set()
        for full_tag in re.findall(r"\\\{(.+?)\\\}", pattern):
            # Ignore .upper and .lower (full_tag: {tag.lower}, tag: {tag})
            if full_tag.endswith((r"\.upper", r"\.lower")):
                tag = full_tag[:-7]
            else:
                tag = full_tag

            old_str = rf"\{{{full_tag}\}}"
            if tag in already_used_tags:
                new_str = rf"(?P={tag})"
            else:
                new_str = rf"(?P<{tag}>[^_{os.sep}]*?)"
                already_used_tags.add(tag)

            pattern = pattern.replace(old_str, new_str, 1)

        # Convert fnmatch wildcards * and [] to regex wildcards
        pattern = pattern.replace(r"\*", ".*?")
        for chars in re.findall(r"\\\[(.*?)\\\]", pattern):
            pattern = pattern.replace(rf"\[{chars}\]", f"[{chars}]")

        return pattern


def _get_multiproduct_filename(attributes: dict, preproc_dir: Path) -> Path:
    """Get ensemble/multi-model filename depending on settings."""
    relevant_keys = [
        "project",
        "dataset",
        "exp",
        "ensemble_statistics",
        "multi_model_statistics",
        "mip",
        "short_name",
    ]

    filename_segments = []
    for key in relevant_keys:
        if key in attributes:
            attribute = attributes[key]
            if isinstance(attribute, (list, tuple)):
                attribute = "-".join(attribute)
            filename_segments.extend(attribute.split("_"))

    # Remove duplicate segments:
    filename_segments = list(dict.fromkeys(filename_segments))

    # Add time period if possible
    if "timerange" in attributes:
        filename_segments.append(
            f"{attributes['timerange'].replace('/', '-')}",
        )

    filename = f"{'_'.join(filename_segments)}.nc"
    return Path(
        preproc_dir,
        attributes["diagnostic"],
        attributes["variable_group"],
        filename,
    )


def _filter_versions_called_latest(
    files: list[LocalFile],
) -> list[LocalFile]:
    """Filter out versions called 'latest' if they are duplicates.

    On compute clusters it is usual to have a symbolic link to the
    latest version called 'latest'. Those need to be skipped in order to
    find valid version names and avoid duplicate results.
    """
    resolved_valid_versions = {
        f.resolve(strict=False)
        for f in files
        if f.facets.get("version") != "latest"
    }
    return [
        f
        for f in files
        if f.facets.get("version") != "latest"
        or f.resolve(strict=False) not in resolved_valid_versions
    ]


def _select_latest_version(files: list[LocalFile]) -> list[LocalFile]:
    """Select only the latest version of files."""

    def filename(file):
        return file.name

    def version(file):
        return file.facets.get("version", "")

    result = []
    for _, group in itertools.groupby(
        sorted(files, key=filename),
        key=filename,
    ):
        duplicates = sorted(group, key=version)
        latest = duplicates[-1]
        result.append(latest)
    return result


GRIB_FORMATS = (".grib2", ".grib", ".grb2", ".grb", ".gb2", ".gb")
"""GRIB file extensions."""


def _get_attr_from_field_coord(
    ncfield: iris.fileformats.cf.CFVariable,
    coord_name: str | None,
    attr: str,
) -> Any:  # noqa: ANN401
    """Get attribute from netCDF field coordinate."""
    if coord_name is not None:
        attrs = ncfield.cf_group[coord_name].cf_attrs()
        attr_val = [value for (key, value) in attrs if key == attr]
        if attr_val:
            return attr_val[0]
    return None


def _restore_lat_lon_units(
    cube: iris.cube.Cube,
    field: iris.fileformats.cf.CFVariable,
    filename: str,  # noqa: ARG001
) -> None:  # pylint: disable=unused-argument
    """Use this callback to restore the original lat/lon units."""
    # Iris chooses to change longitude and latitude units to degrees
    # regardless of value in file, so reinstating file value
    for coord in cube.coords():
        if coord.standard_name in ["longitude", "latitude"]:
            units = _get_attr_from_field_coord(field, coord.var_name, "units")
            if units is not None:
                coord.units = units


class LocalFile(type(Path()), esmvalcore.io.protocol.DataElement):  # type: ignore
    """File on the local filesystem."""

    def prepare(self) -> None:
        """Prepare the data for access."""

    @property
    def facets(self) -> Facets:
        """Facets are key-value pairs that were used to find this data."""
        if not hasattr(self, "_facets"):
            self._facets: Facets = {}
        return self._facets

    @facets.setter
    def facets(self, value: Facets) -> None:
        self._facets = value

    @property
    def attributes(self) -> dict[str, Any]:
        """Attributes are key-value pairs describing the data."""
        if not hasattr(self, "_attributes"):
            msg = (
                "Attributes have not been read yet. Call the `to_iris` method "
                "first to read the attributes from the file."
            )
            raise ValueError(msg)
        return self._attributes

    @attributes.setter
    def attributes(self, value: dict[str, Any]) -> None:
        self._attributes = value

    @property
    def ignore_warnings(self) -> list[dict[str, Any]] | None:
        """Warnings to ignore when loading the data.

        The list should contain :class:`dict`s with keyword arguments that
        will be passed to the :func:`warnings.filterwarnings` function when
        calling the ``to_iris`` method.
        """
        if not hasattr(self, "_ignore_warnings"):
            self._ignore_warnings: list[dict[str, Any]] | None = None
        return self._ignore_warnings

    @ignore_warnings.setter
    def ignore_warnings(self, value: list[dict[str, Any]] | None) -> None:
        self._ignore_warnings = value

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        iris.cube.CubeList
            The loaded data.
        """
        file = Path(self)

        with ignore_warnings_context(self.ignore_warnings):
            # GRIB files need to be loaded with iris.load, otherwise we will
            # get separate (lat, lon) slices for each time step, pressure
            # level, etc.
            if file.suffix in GRIB_FORMATS:
                cubes = iris.load(file, callback=_restore_lat_lon_units)
            else:
                cubes = iris.load_raw(file, callback=_restore_lat_lon_units)

        for cube in cubes:
            cube.attributes.globals["source_file"] = str(file)

        # Cache the attributes.
        self.attributes = copy.deepcopy(dict(cubes[0].attributes.globals))
        return cubes
