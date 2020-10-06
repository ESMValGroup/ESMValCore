"""Data finder module for the ESMValTool."""
# Authors:
# Bouwe Andela (eScience, NL - b.andela@esciencecenter.nl)
# Valeriu Predoi (URead, UK - valeriu.predoi@ncas.ac.uk)
# Mattia Righi (DLR, Germany - mattia.righi@dlr.de)

import fnmatch
import glob
import logging
import os
import re
from pathlib import Path

import iris

logger = logging.getLogger(__name__)


def find_files(dirnames, filenames):
    """Find files matching filenames in dirnames."""
    logger.debug("Looking for files matching %s in %s", filenames, dirnames)

    result = []
    for dirname in dirnames:
        for path, _, files in os.walk(dirname, followlinks=True):
            for filename in filenames:
                matches = fnmatch.filter(files, filename)
                result.extend(os.path.join(path, f) for f in matches)

    return result


def get_start_end_year(filename):
    """Get the start and end year from a file name."""
    stem = Path(filename).stem
    start_year = end_year = None

    # First check for a block of two potential dates separated by _ or -
    daterange = re.findall(r'([0-9]{4,12}[-_][0-9]{4,12})', stem)
    if daterange:
        start_date, end_date = re.findall(r'([0-9]{4,12})', daterange[0])
        start_year = start_date[:4]
        end_year = end_date[:4]
    else:
        # Check for single dates in the filename
        dates = re.findall(r'([0-9]{4,12})', stem)
        if len(dates) == 1:
            start_year = end_year = dates[0][:4]
        elif len(dates) > 1:
            # Check for dates at start or end of filename
            outerdates = re.findall(r'^[0-9]{4,12}|[0-9]{4,12}$', stem)
            if len(outerdates) == 1:
                start_year = end_year = outerdates[0][:4]

    # As final resort, try to get the dates from the file contents
    if start_year is None or end_year is None:
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

    logger.debug("Found start_year %s and end_year %s", start_year, end_year)
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


def _replace_tags(path, variable):
    """Replace tags in the config-developer's file with actual values."""
    path = path.strip('/')
    tlist = re.findall(r'{([^}]*)}', path)
    paths = [path]
    for tag in tlist:
        original_tag = tag
        tag, _, _ = _get_caps_options(tag)

        if tag == 'latestversion':  # handled separately later
            replacewith = '*'
        elif tag in variable:
            replacewith = variable[tag]
        else:
            raise KeyError("Dataset key {} must be specified for {}, check "
                           "your recipe entry".format(tag, variable))

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
    return result


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
    """Resolve the 'latestversion' tag."""
    versions = glob.glob(dirname_template)

    if not versions:
        return dirname_template

    # pick 'latest' if it exists,
    # otherwise take the last sorted == most recent item
    for version in sorted(versions):
        if 'latest' in version:
            break

    return version


def get_output_file(variable, output_file):
    """Return the full path to the output (preprocessed) file."""
    # Join different experiment names
    if isinstance(variable.get('exp'), (list, tuple)):
        variable = dict(variable)
        variable['exp'] = '-'.join(variable['exp'])

    filename = _replace_tags(output_file, variable)[0]

    if variable['frequency'] != 'fx':
        filename += '_{start_year}-{end_year}'.format(**variable)
    filename += '.nc'

    return Path(variable['diagnostic'], variable['variable_group'], filename)


def get_statistic_output_file(variable):
    """Get multi model statistic filename depending on settings."""
    template = '{dataset}_{mip}_{short_name}_{start_year}-{end_year}.nc'
    filename = template.format(**variable)

    return Path(variable['diagnostic'], variable['variable_group'], filename)
