"""Data finder module for the ESMValTool."""
import glob
import logging
import os
import re
from pathlib import Path

import iris

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


def get_start_end_year(filename):
    """Get the start and end year from a file name.

    Examples of allowed dates : 1980, 198001, 19801231,
    1980123123, 19801231T23, 19801231T2359, 19801231T235959,
    19801231T235959Z

    Dates must be surrounded by - or _ or string start or string end
    (after removing filename suffix)

    Look first for two dates separated by - or _, then for one single
    date, and if they are multiple, for one date at start or end
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
    #
    # Next string allows to test that there is an allowed delimiter (or
    # string start or end) close to date range (or to single date)
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
        start_year = daterange.group("year")
        end_year = daterange.group("year_end")
    else:
        # Check for single dates in the filename
        single_date_pattern = context + date_pattern + context
        dates = re.findall(single_date_pattern, stem)
        if len(dates) == 1:
            start_year = end_year = dates[0][0]
        elif len(dates) > 1:
            # Check for dates at start or (exclusive or) end of filename
            start = re.search(r'^' + date_pattern, stem)
            end = re.search(date_pattern + r'$', stem)
            if start and not end:
                start_year = end_year = start.group('year')
            elif end:
                start_year = end_year = end.group('year')

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


def select_files(filenames, start_year, end_year):
    """Select files containing data between start_year and end_year.

    This works for filenames matching *_YYYY*-YYYY*.* or *_YYYY*.*
    """
    selection = []
    for filename in filenames:
        start, end = get_start_end_year(filename)
        if start <= end_year and end >= start_year:
            selection.append(filename)
    return selection


def _replace_tags(paths, variable):
    """Replace tags in the config-developer's file with actual values."""
    if isinstance(paths, str):
        paths = set((paths.strip('/'),))
    else:
        paths = set(path.strip('/') for path in paths)
    tlist = set()
    for path in paths:
        tlist = tlist.union(re.findall(r'{([^}]*)}', path))
    if 'sub_experiment' in variable:
        new_paths = []
        for path in paths:
            new_paths.extend((
                re.sub(r'(\b{ensemble}\b)', r'{sub_experiment}-\1', path),
                re.sub(r'({ensemble})', r'{sub_experiment}-\1', path)
            ))
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
        files = select_files(files, variable['start_year'],
                             variable['end_year'])
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
        outfile += '_{start_year}-{end_year}'.format(**variable)
    outfile += '.nc'
    return outfile


def get_statistic_output_file(variable, preproc_dir):
    """Get multi model statistic filename depending on settings."""
    template = os.path.join(
        preproc_dir,
        '{diagnostic}',
        '{variable_group}',
        '{dataset}_{mip}_{short_name}_{start_year}-{end_year}.nc',
    )

    outfile = template.format(**variable)

    return outfile
