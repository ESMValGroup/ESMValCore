"""Data finder module for the ESMValTool."""
import glob
import logging
import os
import re
from pathlib import Path

import iris
import isodate

from ._config import get_project_config
from .exceptions import RecipeError

logger = logging.getLogger(__name__)


def find_files(dirnames, filenames):
    """Find files matching filenames in dirnames."""
    logger.debug("Looking for files matching %s in %s", filenames, dirnames)

    result = []
    for dirname in dirnames:
        for filename_pattern in filenames:
            pat = os.path.join(dirname, filename_pattern)
            files = glob.glob(pat)
            files.sort()  # sorting makes it easier to see what was found
            result.extend(files)

    return result


def _get_from_pattern(pattern, date_range_pattern, stem, group):
    """Get time, date or datetime from date range patterns in file names."""
    #
    # Next string allows to test that there is an allowed delimiter (or
    # string start or end) close to date range (or to single date)
    start_point = end_point = None
    context = r"(?:^|[-_]|$)"
    #
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


def get_start_end_date(filename):
    """Get the start and end dates as a string from a file name.

    Examples of allowed dates : 1980, 198001, 19801231,
    1980123123, 19801231T23, 19801231T2359, 19801231T235959,
    19801231T235959Z (ISO 8601).

    Dates must be surrounded by - or _ or string start or string end
    (after removing filename suffix).

    Look first for two dates separated by - or _, then for one single
    date, and if they are multiple, for one date at start or end.
    """
    stem = Path(filename).stem
    start_date = end_date = None
    #
    time_pattern = (r"(?P<hour>[0-2][0-9]"
                    r"(?P<minute>[0-5][0-9]"
                    r"(?P<second>[0-5][0-9])?)?Z?)")
    date_pattern = (r"(?P<year>[0-9]{4})"
                    r"(?P<month>[01][0-9]"
                    r"(?P<day>[0-3][0-9]"
                    rf"(T?{time_pattern})?)?)?")
    datetime_pattern = (rf"(?P<datetime>{date_pattern})")
    #
    end_datetime_pattern = datetime_pattern.replace(">", "_end>")
    date_range_pattern = datetime_pattern + r"[-_]" + end_datetime_pattern
    start_date, end_date = _get_from_pattern(datetime_pattern,
                                             date_range_pattern, stem,
                                             'datetime')

    # As final resort, try to get the dates from the file contents
    if (start_date is None or end_date is None) and Path(filename).exists():
        logger.debug("Must load file %s for daterange ", filename)
        cubes = iris.load(filename)

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
        raise ValueError(f'File {filename} dates do not match a recognized'
                         'pattern and time can not be read from the file')

    return start_date, end_date


def dates_to_timerange(start_date, end_date):
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


def _get_timerange_from_years(variable):
    """Build `timerange` tag from tags `start_year` and `end_year`."""
    start_year = variable.get('start_year')
    end_year = variable.get('end_year')
    if start_year and end_year:
        variable['timerange'] = dates_to_timerange(start_year, end_year)
    elif start_year:
        variable['timerange'] = dates_to_timerange(start_year, start_year)
    elif end_year:
        variable['timerange'] = dates_to_timerange(end_year, end_year)
    variable.pop('start_year', None)
    variable.pop('end_year', None)


def get_start_end_year(filename):
    """Get the start and end year from a file name.

    Examples of allowed dates : 1980, 198001, 19801231,
    1980123123, 19801231T23, 19801231T2359, 19801231T235959,
    19801231T235959Z (ISO 8601).

    Dates must be surrounded by - or _ or string start or string end
    (after removing filename suffix).

    Look first for two dates separated by - or _, then for one single
    date, and if they are multiple, for one date at start or end.
    """
    stem = Path(filename).stem
    start_year = end_year = None
    #
    time_pattern = (r"(?P<hour>[0-2][0-9]"
                    r"(?P<minute>[0-5][0-9]"
                    r"(?P<second>[0-5][0-9])?)?Z?)")
    date_pattern = (r"(?P<year>[0-9]{4})"
                    r"(?P<month>[01][0-9]"
                    r"(?P<day>[0-3][0-9]"
                    rf"(T?{time_pattern})?)?)?")
    #
    end_date_pattern = date_pattern.replace(">", "_end>")
    date_range_pattern = date_pattern + r"[-_]" + end_date_pattern
    start_year, end_year = _get_from_pattern(date_pattern, date_range_pattern,
                                             stem, 'year')
    # As final resort, try to get the dates from the file contents
    if (start_year is None or end_year is None) and Path(filename).exists():
        logger.debug("Must load file %s for daterange ", filename)
        cubes = iris.load(filename)

        for cube in cubes:
            logger.debug(cube)
            try:
                time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            start_year = time.cell(0).point.year
            end_year = time.cell(-1).point.year
            break

    if start_year is None or end_year is None:
        raise ValueError(f'File {filename} dates do not match a recognized'
                         'pattern and time can not be read from the file')

    return int(start_year), int(end_year)


def _parse_period(timerange):
    """Parse `timerange` values given as duration periods.

    Sum the duration periods to the `timerange` value given as a
    reference point in order to compute the start and end dates needed
    for file selection.
    """
    start_date = None
    end_date = None
    time_format = None
    datetime_format = (
        isodate.DATE_BAS_COMPLETE + 'T' + isodate.TIME_BAS_COMPLETE)
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
        start_date = str(isodate.datetime_isoformat(
            start_date, format=datetime_format))
        end_date = str(isodate.datetime_isoformat(
            end_date, format=datetime_format))
    elif time_format == isodate.DATE_BAS_COMPLETE:
        start_date = str(
            isodate.date_isoformat(start_date, format=time_format))
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


def select_files(filenames, timerange):
    """Select files containing data between a given timerange.

    If the timerange is given as a period, the file selection
    occurs taking only the years into account.

    Otherwise, the file selection occurs taking into account
    the time resolution of the file.
    """
    selection = []

    for filename in filenames:
        start_date, end_date = _parse_period(timerange)
        start, end = get_start_end_date(filename)

        start_date, start = _truncate_dates(start_date, start)
        end_date, end = _truncate_dates(end_date, end)

        if start <= end_date and end >= start_date:
            selection.append(filename)

    return selection


def _replace_tags(paths, variable):
    """Replace tags in the config-developer's file with actual values."""
    if isinstance(paths, str):
        paths = set((paths.strip('/'), ))
    else:
        paths = set(path.strip('/') for path in paths)
    tlist = set()
    for path in paths:
        tlist = tlist.union(re.findall(r'{([^}]*)}', path))
    if 'sub_experiment' in variable:
        new_paths = []
        for path in paths:
            new_paths.extend(
                (re.sub(r'(\b{ensemble}\b)', r'{sub_experiment}-\1', path),
                 re.sub(r'({ensemble})', r'{sub_experiment}-\1', path)))
            tlist.add('sub_experiment')
        paths = new_paths

    for tag in tlist:
        original_tag = tag
        tag, _, _ = _get_caps_options(tag)

        if tag == 'latestversion':  # handled separately later
            continue
        if tag in variable:
            replacewith = variable[tag]
        else:
            raise RecipeError(f"Dataset key '{tag}' must be specified for "
                              f"{variable}, check your recipe entry")
        paths = _replace_tag(paths, original_tag, replacewith)
    return paths


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


def _resolve_latestversion(dirname_template):
    """Resolve the 'latestversion' tag.

    This implementation avoid globbing on centralized clusters with very
    large data root dirs (i.e. ESGF nodes like Jasmin/DKRZ).
    """
    if '{latestversion}' not in dirname_template:
        return dirname_template

    # Find latest version
    part1, part2 = dirname_template.split('{latestversion}')
    part2 = part2.lstrip(os.sep)
    if os.path.exists(part1):
        versions = os.listdir(part1)
        versions.sort(reverse=True)
        for version in ['latest'] + versions:
            dirname = os.path.join(part1, version, part2)
            if os.path.isdir(dirname):
                return dirname

    return None


def _select_drs(input_type, drs, project):
    """Select the directory structure of input path."""
    cfg = get_project_config(project)
    input_path = cfg[input_type]
    if isinstance(input_path, str):
        return input_path

    structure = drs.get(project, 'default')
    if structure in input_path:
        return input_path[structure]

    raise KeyError(
        'drs {} for {} project not specified in config-developer file'.format(
            structure, project))


ROOTPATH_WARNED = set()


def get_rootpath(rootpath, project):
    """Select the rootpath."""
    for key in (project, 'default'):
        if key in rootpath:
            nonexistent = tuple(p for p in rootpath[key]
                                if not os.path.exists(p))
            if nonexistent and (key, nonexistent) not in ROOTPATH_WARNED:
                logger.warning(
                    "'%s' rootpaths '%s' set in config-user.yml do not exist",
                    key, ', '.join(nonexistent))
                ROOTPATH_WARNED.add((key, nonexistent))
            return rootpath[key]
    raise KeyError('default rootpath must be specified in config-user file')


def _find_input_dirs(variable, rootpath, drs):
    """Return a the full paths to input directories."""
    project = variable['project']

    root = get_rootpath(rootpath, project)
    path_template = _select_drs('input_dir', drs, project)

    dirnames = []
    for dirname_template in _replace_tags(path_template, variable):
        for base_path in root:
            dirname = os.path.join(base_path, dirname_template)
            dirname = _resolve_latestversion(dirname)
            if dirname is None:
                continue
            matches = glob.glob(dirname)
            matches = [match for match in matches if os.path.isdir(match)]
            if matches:
                for match in matches:
                    dirnames.append(match)
            else:
                logger.debug("Skipping non-existent %s", dirname)

    return dirnames


def _get_filenames_glob(variable, drs):
    """Return patterns that can be used to look for input files."""
    path_template = _select_drs('input_file', drs, variable['project'])
    filenames_glob = _replace_tags(path_template, variable)
    return filenames_glob


def _find_input_files(variable, rootpath, drs):
    """Find available input files.

    Return the files, the directory in which they are located in, and
    the file name.
    """
    short_name = variable['short_name']
    variable['short_name'] = variable['original_short_name']
    input_dirs = _find_input_dirs(variable, rootpath, drs)
    filenames_glob = _get_filenames_glob(variable, drs)
    files = find_files(input_dirs, filenames_glob)
    variable['short_name'] = short_name
    return (files, input_dirs, filenames_glob)


def get_input_filelist(variable, rootpath, drs):
    """Return the full path to input files."""
    # change ensemble to fixed r0i0p0 for fx variables
    # this is needed and is not a duplicate effort
    if variable['project'] == 'CMIP5' and variable['frequency'] == 'fx':
        variable['ensemble'] = 'r0i0p0'
    (files, dirnames, filenames) = _find_input_files(variable, rootpath, drs)
    # do time gating only for non-fx variables
    if variable['frequency'] != 'fx':
        files = select_files(
            files,
            variable['timerange'])
    return (files, dirnames, filenames)


def get_output_file(variable, preproc_dir):
    """Return the full path to the output (preprocessed) file."""
    cfg = get_project_config(variable['project'])

    # Join different experiment names
    if isinstance(variable.get('exp'), (list, tuple)):
        variable = dict(variable)
        variable['exp'] = '-'.join(variable['exp'])

    outfile = os.path.join(
        preproc_dir,
        variable['diagnostic'],
        variable['variable_group'],
        _replace_tags(cfg['output_file'], variable)[0],
    )
    if variable['frequency'] != 'fx':
        timerange = variable['timerange'].replace('/', '-')
        outfile += f'_{timerange}'

    outfile += '.nc'
    return outfile


def get_multiproduct_filename(attributes, preproc_dir):
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
    filename_segments.append(
        f"{attributes['timerange'].replace('/', '-')}.nc")

    outfile = os.path.join(
        preproc_dir,
        attributes['diagnostic'],
        attributes['variable_group'],
        '_'.join(filename_segments),
    )

    return outfile
