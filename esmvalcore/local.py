"""Find files on the local filesystem."""
from __future__ import annotations

import itertools
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import iris
import isodate

from .config import CFG
from .config._config import get_project_config
from .exceptions import RecipeError
from .typing import Facets, FacetValue

if TYPE_CHECKING:
    from .esgf import ESGFFile

logger = logging.getLogger(__name__)


def _get_from_pattern(pattern, date_range_pattern, stem, group):
    """Get time, date or datetime from date range patterns in file names."""
    # Next string allows to test that there is an allowed delimiter (or
    # string start or end) close to date range (or to single date)
    start_point = end_point = None
    context = r"(?:^|[-_]|$)"

    # First check for a block of two potential dates
    date_range_pattern_with_context = context + date_range_pattern + context
    daterange = re.search(date_range_pattern_with_context, stem)
    if not daterange:
        # Retry with extended context for CMIP3
        context = r"(?:^|[-_.]|$)"
        date_range_pattern_with_context = (context + date_range_pattern +
                                           context)
        daterange = re.search(date_range_pattern_with_context, stem)

    if daterange:
        start_point = daterange.group(group)
        end_group = '_'.join([group, 'end'])
        end_point = daterange.group(end_group)
    else:
        # Check for single dates in the filename
        single_date_pattern = context + pattern + context
        dates = re.findall(single_date_pattern, stem)
        if len(dates) == 1:
            start_point = end_point = dates[0][0]
        elif len(dates) > 1:
            # Check for dates at start or (exclusive or) end of filename
            start = re.search(r'^' + pattern, stem)
            end = re.search(pattern + r'$', stem)
            if start and not end:
                start_point = end_point = start.group(group)
            elif end:
                start_point = end_point = end.group(group)

    return start_point, end_point


def _get_start_end_date(
        file: str | Path | LocalFile | ESGFFile) -> tuple[str, str]:
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
    if hasattr(file, 'name'):  # Path, LocalFile, ESGFFile
        stem = Path(file.name).stem
    else:  # str
        stem = Path(file).stem

    start_date = end_date = None

    # Build regex
    time_pattern = (r"(?P<hour>[0-2][0-9]"
                    r"(?P<minute>[0-5][0-9]"
                    r"(?P<second>[0-5][0-9])?)?Z?)")
    date_pattern = (r"(?P<year>[0-9]{4})"
                    r"(?P<month>-?[01][0-9]"
                    r"(?P<day>-?[0-3][0-9]"
                    rf"(T?{time_pattern})?)?)?")
    datetime_pattern = (rf"(?P<datetime>{date_pattern})")
    end_datetime_pattern = datetime_pattern.replace(">", "_end>")

    # Dates can either be delimited by '-', '_', or '_cat_' (the latter for
    # CMIP3)
    date_range_pattern = (datetime_pattern + r"[-_](?:cat_)?" +
                          end_datetime_pattern)

    # Find dates using the regex
    start_date, end_date = _get_from_pattern(datetime_pattern,
                                             date_range_pattern, stem,
                                             'datetime')

    # As final resort, try to get the dates from the file contents
    if ((start_date is None or end_date is None)
            and isinstance(file, (str, Path)) and Path(file).exists()):
        logger.debug("Must load file %s for daterange ", file)
        cubes = iris.load(file)

        for cube in cubes:
            logger.debug(cube)
            try:
                time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            start_date = isodate.date_isoformat(
                time.cell(0).point, format=isodate.isostrf.DATE_BAS_COMPLETE)

            end_date = isodate.date_isoformat(
                time.cell(-1).point, format=isodate.isostrf.DATE_BAS_COMPLETE)
            break

    if start_date is None or end_date is None:
        raise ValueError(
            f"File {file} datetimes do not match a recognized pattern and "
            f"time coordinate can not be read from the file")

    # Remove potential '-' characters from datetimes
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    return start_date, end_date


def _get_start_end_year(
        file: str | Path | LocalFile | ESGFFile) -> tuple[int, int]:
    """Get the start and end year as int from a file name.

    See :func:`_get_start_end_date`.
    """
    (start_date, end_date) = _get_start_end_date(file)
    return (int(start_date[:4]), int(end_date[:4]))


def _dates_to_timerange(start_date, end_date):
    """Convert ``start_date`` and ``end_date`` to ``timerange``.

    Note
    ----
    This function ensures that dates in years format follow the pattern YYYY
    (i.e., that they have at least 4 digits). Other formats, such as  wildcards
    (``'*'``) and relative time ranges (e.g., ``'P6Y'``) are used unchanged.

    Parameters
    ----------
    start_date: int or str
        Start date.
    end_date: int or str
        End date.

    Returns
    -------
    str
        ``timerange`` in the form ``'start_date/end_date'``.
    """
    start_date = str(start_date)
    end_date = str(end_date)

    # Pad years with 0s if not wildcard or relative time range
    if start_date != '*' and not start_date.startswith('P'):
        start_date = start_date.zfill(4)
    if end_date != '*' and not end_date.startswith('P'):
        end_date = end_date.zfill(4)

    return f'{start_date}/{end_date}'


def _replace_years_with_timerange(variable):
    """Set `timerange` tag from tags `start_year` and `end_year`."""
    start_year = variable.get('start_year')
    end_year = variable.get('end_year')
    if start_year and end_year:
        variable['timerange'] = _dates_to_timerange(start_year, end_year)
    elif start_year:
        variable['timerange'] = _dates_to_timerange(start_year, start_year)
    elif end_year:
        variable['timerange'] = _dates_to_timerange(end_year, end_year)
    variable.pop('start_year', None)
    variable.pop('end_year', None)


def _parse_period(timerange):
    """Parse `timerange` values given as duration periods.

    Sum the duration periods to the `timerange` value given as a
    reference point in order to compute the start and end dates needed
    for file selection.
    """
    start_date = None
    end_date = None
    time_format = None
    datetime_format = (isodate.DATE_BAS_COMPLETE + 'T' +
                       isodate.TIME_BAS_COMPLETE)
    if timerange.split('/')[0].startswith('P'):
        try:
            end_date = isodate.parse_datetime(timerange.split('/')[1])
            time_format = datetime_format
        except isodate.ISO8601Error:
            end_date = isodate.parse_date(timerange.split('/')[1])
            time_format = isodate.DATE_BAS_COMPLETE
        delta = isodate.parse_duration(timerange.split('/')[0])
        start_date = end_date - delta
    elif timerange.split('/')[1].startswith('P'):
        try:
            start_date = isodate.parse_datetime(timerange.split('/')[0])
            time_format = datetime_format
        except isodate.ISO8601Error:
            start_date = isodate.parse_date(timerange.split('/')[0])
            time_format = isodate.DATE_BAS_COMPLETE
        delta = isodate.parse_duration(timerange.split('/')[1])
        end_date = start_date + delta

    if time_format == datetime_format:
        start_date = str(
            isodate.datetime_isoformat(start_date, format=datetime_format))
        end_date = str(
            isodate.datetime_isoformat(end_date, format=datetime_format))
    elif time_format == isodate.DATE_BAS_COMPLETE:
        start_date = str(isodate.date_isoformat(start_date,
                                                format=time_format))
        end_date = str(isodate.date_isoformat(end_date, format=time_format))

    if start_date is None and end_date is None:
        start_date = timerange.split('/')[0]
        end_date = timerange.split('/')[1]

    return start_date, end_date


def _truncate_dates(date, file_date):
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
    date = re.sub("[^0-9]", '', date)
    file_date = re.sub("[^0-9]", '', file_date)
    if len(date) < len(file_date):
        file_date = file_date[0:len(date)]
    elif len(date) > len(file_date):
        date = date[0:len(file_date)]

    return int(date), int(file_date)


def _select_files(filenames, timerange):
    """Select files containing data between a given timerange.

    If the timerange is given as a period, the file selection occurs
    taking only the years into account.

    Otherwise, the file selection occurs taking into account the time
    resolution of the file.
    """
    if '*' in timerange:
        # TODO: support * combined with a period
        return filenames

    selection = []

    for filename in filenames:
        start_date, end_date = _parse_period(timerange)
        start, end = _get_start_end_date(filename)

        start_date, end = _truncate_dates(start_date, end)
        end_date, start = _truncate_dates(end_date, start)
        if start <= end_date and end >= start_date:
            selection.append(filename)

    return selection


def _replace_tags(
    paths: Union[str, list[str]],
    variable: Facets,
) -> list[Path]:
    """Replace tags in the config-developer's file with actual values."""
    if isinstance(paths, str):
        pathset = set((paths.strip('/'), ))
    else:
        pathset = set(path.strip('/') for path in paths)
    tlist: set[str] = set()
    for path in pathset:
        tlist = tlist.union(re.findall(r'{([^}]*)}', path))
    if 'sub_experiment' in variable:
        new_paths: set[str] = set()
        for path in pathset:
            new_paths.update(
                (re.sub(r'(\b{ensemble}\b)', r'{sub_experiment}-\1', path),
                 re.sub(r'({ensemble})', r'{sub_experiment}-\1', path)))
            tlist.add('sub_experiment')
        pathset = new_paths

    for tag in tlist:
        original_tag = tag
        tag, _, _ = _get_caps_options(tag)

        if tag in variable:
            replacewith = variable[tag]
        elif tag == 'version':
            replacewith = '*'
        else:
            raise RecipeError(f"Dataset key '{tag}' must be specified for "
                              f"{variable}, check your recipe entry")
        pathset = _replace_tag(pathset, original_tag, replacewith)
    return [Path(p) for p in pathset]


def _replace_tag(paths, tag, replacewith):
    """Replace tag by replacewith in paths."""
    _, lower, upper = _get_caps_options(tag)
    result = []
    if isinstance(replacewith, (list, tuple)):
        for item in replacewith:
            result.extend(_replace_tag(paths, tag, item))
    else:
        text = _apply_caps(str(replacewith), lower, upper)
        result.extend(p.replace('{' + tag + '}', text) for p in paths)
    return list(set(result))


def _get_caps_options(tag):
    lower = False
    upper = False
    if tag.endswith('.lower'):
        lower = True
        tag = tag[0:-6]
    elif tag.endswith('.upper'):
        upper = True
        tag = tag[0:-6]
    return tag, lower, upper


def _apply_caps(original, lower, upper):
    if lower:
        return original.lower()
    if upper:
        return original.upper()
    return original


def _select_drs(input_type, project):
    """Select the directory structure of input path."""
    cfg = get_project_config(project)
    input_path = cfg[input_type]
    if isinstance(input_path, str):
        return input_path

    structure = CFG['drs'].get(project, 'default')
    if structure in input_path:
        return input_path[structure]

    raise KeyError(
        'drs {} for {} project not specified in config-developer file'.format(
            structure, project))


_ROOTPATH_WARNED = set()


def _get_rootpath(project):
    """Select the rootpath."""
    rootpath = CFG['rootpath']
    for key in (project, 'default'):
        if key in rootpath:
            nonexistent = tuple(p for p in rootpath[key]
                                if not os.path.exists(p))
            if nonexistent and (key, nonexistent) not in _ROOTPATH_WARNED:
                logger.warning(
                    "'%s' rootpaths '%s' set in config-user.yml do not exist",
                    key, ', '.join(str(p) for p in nonexistent))
                _ROOTPATH_WARNED.add((key, nonexistent))
            return rootpath[key]
    raise KeyError(f'The "{project}" option is missing from the "rootpath" '
                   'section in the config-user.yml file.')


def _get_globs(variable):
    """Compose the globs that will be used to look for files."""
    project = variable['project']

    rootpaths = _get_rootpath(project)

    dirname_template = _select_drs('input_dir', project)
    dirname_globs = _replace_tags(dirname_template, variable)

    filename_template = _select_drs('input_file', project)
    filename_globs = _replace_tags(filename_template, variable)

    globs = sorted(r / d / f for r in rootpaths for d in dirname_globs
                   for f in filename_globs)
    return globs


def _get_input_filelist(variable):
    """Return the full path to input files."""
    variable = dict(variable)
    if 'original_short_name' in variable:
        variable['short_name'] = variable['original_short_name']

    globs = _get_globs(variable)
    logger.debug(
        "Looking for files matching:\n%s",
        "\n".join(str(g) for g in globs),
    )

    files = list(Path(file) for glob_ in globs for file in glob(str(glob_)))
    files.sort()  # sorting makes it easier to see what was found

    if 'timerange' in variable:
        files = _select_files(files, variable['timerange'])

    return files, globs


def _get_output_file(variable: dict[str, Any], preproc_dir: Path) -> Path:
    """Return the full path to the output (preprocessed) file."""
    cfg = get_project_config(variable['project'])

    # Join different experiment names
    if isinstance(variable.get('exp'), (list, tuple)):
        variable = dict(variable)
        variable['exp'] = '-'.join(variable['exp'])
    outfile = _replace_tags(cfg['output_file'], variable)[0]
    if 'timerange' in variable:
        timerange = variable['timerange'].replace('/', '-')
        outfile = Path(f'{outfile}_{timerange}')
    outfile = Path(f"{outfile}.nc")
    return Path(
        preproc_dir,
        variable.get('diagnostic', ''),
        variable.get('variable_group', ''),
        outfile,
    )


def _get_multiproduct_filename(attributes: dict, preproc_dir: Path) -> Path:
    """Get ensemble/multi-model filename depending on settings."""
    relevant_keys = [
        'project', 'dataset', 'exp', 'ensemble_statistics',
        'multi_model_statistics', 'mip', 'short_name'
    ]

    filename_segments = []
    for key in relevant_keys:
        if key in attributes:
            attribute = attributes[key]
            if isinstance(attribute, (list, tuple)):
                attribute = '-'.join(attribute)
            filename_segments.extend(attribute.split('_'))

    # Remove duplicate segments:
    filename_segments = list(dict.fromkeys(filename_segments))

    # Add period and extension
    filename_segments.append(f"{attributes['timerange'].replace('/', '-')}.nc")

    outfile = Path(
        preproc_dir,
        attributes['diagnostic'],
        attributes['variable_group'],
        '_'.join(filename_segments),
    )

    return outfile


def _path2facets(path: Path, drs: str) -> dict[str, str]:
    """Extract facets from a path using a DRS like '{facet1}/{facet2}'."""
    keys = []
    for key in re.findall(r'{(.*?)}[^-]', f'{drs} '):
        key = key.split('.')[0]  # Remove trailing .lower and .upper
        keys.append(key)
    start, end = -len(keys) - 1, -1
    values = path.parts[start:end]
    facets = {
        key: values[idx]
        for idx, key in enumerate(keys) if "{" not in key
    }

    if len(facets) != len(keys):
        # Extract hyphen separated facet: {facet1}-{facet2},
        # where facet1 is already known.
        for idx, key in enumerate(keys):
            if key not in facets:
                facet1, facet2 = key.split("}-{")
                facets[facet2] = values[idx].replace(f'{facets[facet1]}-', '')

    return facets


def _filter_versions_called_latest(
        files: list['LocalFile']) -> list['LocalFile']:
    """Filter out versions called 'latest' if they are duplicates.

    On compute clusters it is usual to have a symbolic link to the
    latest version called 'latest'. Those need to be skipped in order to
    find valid version names and avoid duplicate results.
    """
    resolved_valid_versions = {
        f.resolve(strict=False)
        for f in files if f.facets.get('version') != 'latest'
    }
    return [
        f for f in files if f.facets.get('version') != 'latest' or f.resolve(
            strict=False) not in resolved_valid_versions
    ]


def _select_latest_version(files: list['LocalFile']) -> list['LocalFile']:
    """Select only the latest version of files."""

    def filename(file):
        return file.name

    def version(file):
        return file.facets.get('version', '')

    result = []
    for _, group in itertools.groupby(sorted(files, key=filename),
                                      key=filename):
        duplicates = sorted(group, key=version)
        latest = duplicates[-1]
        result.append(latest)
    return result


def find_files(
    *,
    debug: bool = False,
    **facets: FacetValue,
) -> Union[list[LocalFile], tuple[list[LocalFile], list[Path]]]:
    """Find files on the local filesystem.

    The directories that are searched for files are defined in
    :data:`esmvalcore.config.CFG` under the ``'rootpath'`` key using the
    directory structure defined under the ``'drs'`` key.
    If ``esmvalcore.config.CFG['rootpath']`` contains a key that matches the
    value of the ``project`` facet, those paths will be used. If there is no
    project specific key, the directories in
    ``esmvalcore.config.CFG['rootpath']['default']`` will be searched.

    See :ref:`findingdata` for extensive instructions on configuring ESMValCore
    so it can find files locally.

    Parameters
    ----------
    debug
        When debug is set to :obj:`True`, the function will return a tuple
        with the first element containing the files that were found
        and the second element containing the :func:`glob.glob` patterns that
        were used to search for files.
    **facets
        Facets used to search for files. An ``'*'`` can be used to match
        any value. By default, only the latest version of a file will
        be returned. To select all versions use ``version='*'``. It is also
        possible to specify multiple values for a facet, e.g.
        ``exp=['historical', 'ssp585']`` will match any file that belongs
        to either the historical or ssp585 experiment.
        The ``timerange`` facet can be specified in `ISO 8601 format
        <https://en.wikipedia.org/wiki/ISO_8601>`__.

    Note
    ----
    A value of ``timerange='*'`` is supported, but combining a ``'*'`` with
    a time or period :ref:`as supported in the recipe <datasets>` is currently
    not supported and will return all found files.

    Examples
    --------
    Search for files containing surface air temperature from any CMIP6 model
    for the historical experiment:

    >>> esmvalcore.local.find_files(
    ...     project='CMIP6',
    ...     activity='CMIP',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     exp='historical',
    ...     dataset='*',
    ...     ensemble='*',
    ...     grid='*',
    ...     institute='*',
    ... )  # doctest: +SKIP
    [LocalFile('/home/bandela/climate_data/CMIP6/CMIP/BCC/BCC-ESM1/historical/r1i1p1f1/Amon/tas/gn/v20181214/tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc')]

    Returns
    -------
    list[LocalFile]
        The files that were found.
    """  # pylint: disable=line-too-long
    filenames, globs = _get_input_filelist(facets)
    drs = _select_drs('input_dir', facets['project'])
    if isinstance(drs, list):
        # Not sure how to handle a list of DRSs
        drs = ''
    files = []
    filter_latest = False
    for filename in filenames:
        file = LocalFile(filename)
        file.facets.update(_path2facets(file, drs))
        if file.facets.get('version') == 'latest':
            filter_latest = True
        files.append(file)

    if filter_latest:
        files = _filter_versions_called_latest(files)

    if 'version' not in facets:
        files = _select_latest_version(files)

    if debug:
        return files, globs
    return files


class LocalFile(type(Path())):  # type: ignore
    """File on the local filesystem."""

    @property
    def facets(self) -> Facets:
        """Facets describing the file.

        Note
        ----
        When using :func:`find_files`, facets are read from the directory
        structure. Facets stored in filenames are not yet supported.
        """
        if not hasattr(self, '_facets'):
            self._facets: Facets = {}
        return self._facets

    @facets.setter
    def facets(self, value: Facets):
        self._facets = value
