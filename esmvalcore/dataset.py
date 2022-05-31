import copy
import logging
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
from .cmor.table import CMOR_TABLES
from .exceptions import InputFilesNotFound, RecipeError
from .preprocessor import add_fx_variables, preprocess

logger = logging.getLogger(__name__)


class Dataset:

    def __init__(self, **facets):

        self.facets = copy.deepcopy(facets)
        self.ancillaries = []

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.facets == other.facets
                and self.ancillaries == other.ancillaries)

    def __repr__(self):
        return repr(self.facets)

    def add_ancillary(self, **facets):
        _augment(facets, self.facets)
        self.ancillaries.append(Dataset(**facets))

    def augment_facets(self, session):
        """Augment facets."""
        self._augment_facets(session)
        for ancillary in self.ancillaries:
            ancillary._augment_facets(session)

    def _augment_facets(self, session):
        _add_extra_facets(self.facets, session['extra_facets_dir'])
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
        _update_timerange(self, session)

    def find_files(self, session, debug=False):
        """Find files."""
        self._find_files(session, debug)
        for ancillary in self.ancillaries:
            ancillary._find_files(session, debug)

    def _find_files(self, session, debug):
        facets = dict(self.facets)
        if facets['frequency'] != 'fx':
            start_year, end_year = _parse_period(facets['timerange'])

            start_year = int(str(start_year[0:4]))
            end_year = int(str(end_year[0:4]))

            facets['start_year'] = start_year
            facets['end_year'] = end_year
        (input_files, dirnames,
         filenames) = get_input_filelist(facets,
                                         rootpath=session['rootpath'],
                                         drs=session['drs'])

        # Set up downloading from ESGF if requested.
        if (not session['offline']
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
                    local_copy = file.local_file(session['download_dir'])
                    if local_copy.name not in local_files:
                        input_files.append(str(local_copy))

                dirnames.append('ESGF:')

        self.files = input_files
        # TODO: should this function return the files?
        if debug:
            return (input_files, dirnames, filenames)
        return input_files

    def load(self, session):
        """Load dataset."""
        check_level = session['check_level']
        preproc_dir = session.preproc_dir
        cube = self._load(preproc_dir, check_level)
        fx_cubes = []
        for ancillary_dataset in self.ancillaries:
            if ancillary_dataset.files:
                fx_cube = ancillary_dataset._load(preproc_dir, check_level)
                fx_cubes.append(fx_cube)
        add_fx_variables(cube, fx_cubes)
        return cube

    def _load(self, preproc_dir, check_level):
        """Load self.files into an iris cube and return it."""
        output_file = get_output_file(self.facets, preproc_dir)
        settings = {
            'fix_file': {
                'output_dir': f"{output_file.with_suffix('')}_fixed",
                **self.facets,
            },
            'load': {},
            'fix_metadata': {
                'check_level': check_level,
                **self.facets,
            },
            'concatenate': {},
            'fix_data': {
                'check_level': check_level,
                **self.facets,
            },
            'cmor_check_metadata': {
                'check_level': check_level,
                'cmor_table': self.facets['project'],
                'mip': self.facets['mip'],
                'frequency': self.facets['frequency'],
                'short_name': self.facets['short_name'],
            },
            'cmor_check_data': {
                'check_level': check_level,
                'cmor_table': self.facets['project'],
                'mip': self.facets['mip'],
                'frequency': self.facets['frequency'],
                'short_name': self.facets['short_name'],
            },
        }
        if 'timerange' in self.facets and self.facets['frequency'] != 'fx':
            settings['clip_timerange'] = {
                'timerange': self.facets['timerange'],
            }

        result = self.files
        for step, args in settings.items():
            result = preprocess(result, step, input_files=self.files, **args)
        cube = result[0]
        return cube


def _format_facets(facets):
    """Format facets into a kind of human readable string."""
    keys = (
        'project',
        'dataset',
        'rcm_version',
        'driver',
        'domain',
        'mip',
        'exp',
        'ensemble',
        'grid',
        'short_name',
    )
    return ", ".join(facets[k] for k in keys if k in facets)


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


def _update_timerange(dataset: Dataset, session):
    """Update wildcards in timerange with found datetime values.

    If the timerange is given as a year, it ensures it's formatted as a
    4-digit value (YYYY).
    """
    if 'timerange' not in dataset.facets:
        return

    timerange = dataset.facets.get('timerange')
    check.valid_time_selection(timerange)

    if '*' in timerange:
        files = dataset.find_files(session)
        if not files:
            if not session.get('offline', True):
                msg = (
                    " Please note that automatic download is not supported "
                    "with indeterminate time ranges at the moment. Please use "
                    "a concrete time range (i.e., no wildcards '*') in your "
                    "recipe or run ESMValTool with --offline=True.")
            else:
                msg = ""
            raise InputFilesNotFound(
                f"Missing data for: {_format_facets(dataset.facets)}"
                f"Cannot determine time range '{timerange}'.{msg}")

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

    dataset.facets['timerange'] = timerange


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
                if facets['project'] == 'obs4mips':
                    logger.warning(
                        "Correcting capitalization, project 'obs4mips' "
                        "should be written as 'obs4MIPs'")
                    facets['project'] = 'obs4MIPs'
                for facets0 in _expand_tag(facets, 'ensemble'):
                    for facets1 in _expand_tag(facets0, 'sub_experiment'):
                        facets1['recipe_dataset_index'] = idx
                        idx += 1
                        ancillaries = facets1.pop('ancillary_variables', [])
                        dataset = Dataset(**facets1)
                        for ancillary_facets in ancillaries:
                            if isinstance(ancillary_facets, str):
                                ancillary_facets = {
                                    'short_name': ancillary_facets
                                }
                            dataset.add_ancillary(**ancillary_facets)

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
