import copy
import logging
import os.path
import re
from pathlib import Path

from . import _recipe_checks as check
from . import esgf
from ._config import get_activity, get_extra_facets, get_institutes
from ._data_finder import (
    _get_timerange_from_years,
    _parse_period,
    dates_to_timerange,
    get_input_filelist,
    get_output_file,
    get_start_end_date,
)
from ._recipe_checks import RecipeError
from .cmor.check import CheckLevels, cmor_check_data, cmor_check_metadata
from .cmor.fix import fix_data, fix_file, fix_metadata
from .cmor.table import CMOR_TABLES
from .exceptions import InputFilesNotFound
from .preprocessor._io import concatenate, concatenate_callback, load
from .preprocessor._time import clip_timerange

logger = logging.getLogger(__name__)


def _augment(base, update):
    """Update dict base with values from dict update."""
    for key in update:
        if key not in base:
            base[key] = update[key]


def _add_cmor_info(facets, override=False):
    """Add information from CMOR tables to facets."""
    # Copy the following keys from CMOR table
    cmor_keys = [
        'standard_name',
        'long_name',
        'units',
        'modeling_realm',
        'frequency',
    ]
    project = facets['project']
    mip = facets['mip']
    short_name = facets['short_name']
    derive = facets.get('derive', False)
    table = CMOR_TABLES.get(project)
    if table:
        table_entry = table.get_variable(mip, short_name, derive)
    else:
        table_entry = None
    if table_entry is None:
        raise RecipeError(
            f"Unable to load CMOR table (project) '{project}' for variable "
            f"'{short_name}' with mip '{mip}'")
    facets['original_short_name'] = table_entry.short_name
    for key in cmor_keys:
        if key not in facets or override:
            value = getattr(table_entry, key, None)
            if value is not None:
                facets[key] = value
            else:
                logger.debug(
                    "Failed to add key %s to variable %s from CMOR table", key,
                    facets)

    # Check that keys are available
    check.variable(facets, required_keys=cmor_keys)


def _add_extra_facets(facets, extra_facets_dir):
    """Add extra facets from configuration files."""
    extra_facets = get_extra_facets(facets["project"], facets["dataset"],
                                    facets["mip"], facets["short_name"],
                                    extra_facets_dir)
    _augment(facets, extra_facets)


def _update_timerange(facets, config_user):
    """Update wildcards in timerange with found datetime values.

    If the timerange is given as a year, it ensures it's formatted as a
    4-digit value (YYYY).
    """
    if 'timerange' not in facets:
        return

    timerange = facets.get('timerange')
    check.valid_time_selection(timerange)

    if '*' in timerange:
        (files, _, _) = _find_input_files(facets, config_user['rootpath'],
                                          config_user['drs'])
        if not files:
            if not config_user.get('offline', True):
                msg = (
                    " Please note that automatic download is not supported "
                    "with indeterminate time ranges at the moment. Please use "
                    "a concrete time range (i.e., no wildcards '*') in your "
                    "recipe or run ESMValTool with --offline=True.")
            else:
                msg = ""
            raise InputFilesNotFound(
                f"Missing data for {facets['alias']}: "
                f"{facets['short_name']}. Cannot determine indeterminate "
                f"time range '{timerange}'.{msg}")

        intervals = [get_start_end_date(name) for name in files]

        min_date = min(interval[0] for interval in intervals)
        max_date = max(interval[1] for interval in intervals)

        if timerange == '*':
            timerange = f'{min_date}/{max_date}'
        if '*' in timerange.split('/')[0]:
            timerange = timerange.replace('*', min_date)
        if '*' in timerange.split('/')[1]:
            timerange = timerange.replace('*', max_date)

    # Make sure that years are in format YYYY
    (start_date, end_date) = timerange.split('/')
    timerange = dates_to_timerange(start_date, end_date)
    check.valid_time_selection(timerange)

    facets['timerange'] = timerange


class Dataset:

    def __init__(self, **facets):

        self.facets = facets

    def __eq__(self, other):
        return isinstance(other,
                          self.__class__) and self.facets == other.facets

    def __repr__(self):
        return repr(self.facets)

    def augment_facets(self, config_user):
        """Augment facets."""
        _add_extra_facets(self.facets, config_user['extra_facets_dir'])
        if 'institute' not in self.facets:
            institute = get_institutes(self.facets)
            if institute:
                self.facets['institute'] = institute
        if 'activity' not in self.facets:
            activity = get_activity(self.facets)
            if activity:
                self.facets['activity'] = activity
        _add_cmor_info(self.facets)
        _get_timerange_from_years(self.facets)
        _update_timerange(self.facets, config_user)
        self.facets['filename'] = get_output_file(self.facets,
                                                  config_user['preproc_dir'])

    def find_files(self, config_user, debug=False):
        """Get the input files for a single dataset (locally and via download)."""
        facets = dict(self.facets)
        if facets['frequency'] != 'fx':
            start_year, end_year = _parse_period(facets['timerange'])

            start_year = int(str(start_year[0:4]))
            end_year = int(str(end_year[0:4]))

            facets['start_year'] = start_year
            facets['end_year'] = end_year
        (input_files, dirnames,
         filenames) = get_input_filelist(facets,
                                         rootpath=config_user['rootpath'],
                                         drs=config_user['drs'])

        # Set up downloading from ESGF if requested.
        if (not config_user['offline']
                and facets['project'] in esgf.facets.FACETS):
            try:
                check.data_availability(
                    input_files,
                    facets,
                    dirnames,
                    filenames,
                    log=False,
                )
            except RecipeError:
                # Only look on ESGF if files are not available locally.
                local_files = set(Path(f).name for f in input_files)
                search_result = esgf.find_files(**facets)
                for file in search_result:
                    local_copy = file.local_file(config_user['download_dir'])
                    if local_copy.name not in local_files:
                        input_files.append(str(local_copy))

                dirnames.append('ESGF:')

        self.files = input_files
        if debug:
            return (input_files, dirnames, filenames)
        return input_files

    def load(self, check_level=CheckLevels.DEFAULT):
        """Load dataset.

        """

        cubes = []
        for filename in self.files:
            fix_dir = os.path.splitext(self.facets['filename'])[0] + '_fixed'
            filename = fix_file(filename, output_dir=fix_dir, **self.facets)

            file_cubes = load(filename, callback=concatenate_callback)
            cubes.extend(file_cubes)

        cubes = fix_metadata(cubes, check_level=check_level, **self.facets)

        cube = concatenate(cubes)

        cube = cmor_check_metadata(
            cube,
            cmor_table=self.facets['project'],
            mip=self.facets['mip'],
            short_name=self.facets['short_name'],
            frequency=self.facets['frequency'],
            check_level=check_level,
        )

        if 'timerange' in self.facets and self.facets['frequency'] != 'fx':
            cube = clip_timerange(cube, self.facets['timerange'])

        cube = fix_data(cube, check_level=check_level, **self.facets)

        cube = cmor_check_data(
            cube,
            cmor_table=self.facets['project'],
            mip=self.facets['mip'],
            short_name=self.facets['short_name'],
            frequency=self.facets['frequency'],
            check_level=check_level,
        )
        # TODO: add fx variables with `add_fx_variables`

        return cube


def _expand_tag(facets, input_tag):
    """Expand tags such as ensemble members or stardates to multiple datasets.

    Expansion only supports ensembles defined as strings, not lists.
    """
    expanded = []
    regex = re.compile(r'\(\d+:\d+\)')

    def expand_tag(facets_, input_tag):
        tag = facets_.get(input_tag, "")
        match = regex.search(tag)
        if match:
            start, end = match.group(0)[1:-1].split(':')
            for i in range(int(start), int(end) + 1):
                expand = copy.deepcopy(facets_)
                expand[input_tag] = regex.sub(str(i), tag, 1)
                expand_tag(expand, input_tag)
        else:
            expanded.append(facets_)

    tag = facets.get(input_tag, "")
    if isinstance(tag, (list, tuple)):
        for elem in tag:
            if regex.search(elem):
                raise RecipeError(
                    f"In variable {facets}: {input_tag} expansion "
                    f"cannot be combined with {input_tag} lists")
        expanded.append(facets)
    else:
        expand_tag(facets, input_tag)

    return expanded


def datasets_from_recipe(recipe):

    datasets = []

    for diagnostic in recipe['diagnostics']:
        for variable_group in recipe['diagnostics'][diagnostic].get(
                'variables', {}):
            # Read variable from recipe
            recipe_variable = recipe['diagnostics'][diagnostic]['variables'][
                variable_group]
            if recipe_variable is None:
                recipe_variable = {}
            # Read datasets from recipe
            recipe_datasets = (recipe.get('datasets', []) +
                               recipe['diagnostics'][diagnostic].get(
                                   'additional_datasets', []) +
                               recipe_variable.get('additional_datasets', []))

            idx = 0
            for recipe_dataset in recipe_datasets:
                facets = copy.deepcopy(recipe_variable)
                facets.pop('additional_datasets', None)
                facets.update(copy.deepcopy(recipe_dataset))
                facets['diagnostic'] = diagnostic
                facets['variable_group'] = variable_group
                if 'short_name' not in facets:
                    facets['short_name'] = variable_group

                for facets0 in _expand_tag(facets, 'ensemble'):
                    for facets1 in _expand_tag(facets0, 'sub_experiment'):
                        facets1['recipe_dataset_index'] = idx
                        idx += 1
                        dataset = Dataset(**facets1)
                        datasets.append(dataset)
    return datasets


def datasets_to_recipe(datasets):

    diagnostics = {}

    for dataset in datasets:
        facets = dict(dataset.facets)
        facets.pop('recipe_dataset_index', None)
        diagnostic = facets.pop('diagnostic')
        if diagnostic not in diagnostics:
            diagnostics[diagnostic] = {'variables': {}}
        variables = diagnostics[diagnostic]['variables']
        variable_group = facets.pop('variable_group')
        if variable_group not in variables:
            variables[variable_group] = {'additional_datasets': []}
        variables[variable_group]['additional_datasets'].append(facets)

    # TODO: make recipe look nice
    # - move identical facets from dataset to variable
    # - deduplicate by moving datasets up from variable to diagnostic to recipe
    # - remove variable_group if the same as short_name
    # - remove automatically added facets

    # TODO: integrate with existing recipe

    recipe = {'diagnostics': diagnostics}
    return recipe
