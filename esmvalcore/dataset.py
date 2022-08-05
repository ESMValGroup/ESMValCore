import copy
import logging
import pprint
import re
import textwrap
from itertools import groupby
from pathlib import Path

from iris.cube import Cube

from . import esgf
from ._config import Session, get_activity, get_extra_facets, get_institutes
from ._data_finder import (
    _get_timerange_from_years,
    dates_to_timerange,
    get_input_filelist,
    get_output_file,
    get_start_end_date,
)
from ._recipe_checks import _format_facets
from ._recipe_checks import data_availability as check_data_availability
from ._recipe_checks import datasets as check_datasets
from ._recipe_checks import valid_time_selection as check_valid_time_selection
from ._recipe_checks import variable as check_variable
from .cmor.table import _get_facets_from_cmor_table
from .esgf import ESGFFile
from .exceptions import InputFilesNotFound, RecipeError
from .preprocessor import preprocess
from .preprocessor._io import DATASET_KEYS

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = (
    'short_name',
    'mip',
    'dataset',
    'project',
)

__all__ = [
    'Dataset',
    'datasets_from_recipe',
    'datasets_to_recipe',
]


class Dataset:

    def __init__(self, **facets):

        self.facets = {}
        self.ancillaries = []

        self._persist = set()
        self._session = None
        self._files = None
        self._files_debug = (None, None)

        for key, value in facets.items():
            self.set_facet(key, copy.deepcopy(value), persist=True)

    @property
    def minimal_facets(self):
        return {k: v for k, v in self.facets.items() if k in self._persist}

    def copy(self, **facets):
        new = self.__class__()
        new.session = self._session
        for key, value in self.facets.items():
            new.set_facet(key, copy.deepcopy(value), key in self._persist)
        for key, value in facets.items():
            new.set_facet(key, copy.deepcopy(value))
        for ancillary in self.ancillaries:
            # The short_name and mip of the ancillary variable are probably
            # different from the main variable, so don't copy those facets.
            skip = ('short_name', 'mip')
            ancillary_facets = {k: facets[k] for k in facets if k not in skip}
            new_ancillary = ancillary.copy(**ancillary_facets)
            new.ancillaries.append(new_ancillary)
        return new

    def __eq__(self, other):
        try:
            other_session = other.session
        except ValueError:
            other_session = None
        return (isinstance(other, self.__class__)
                and self._session == other_session
                and self.facets == other.facets
                and self.ancillaries == other.ancillaries)

    def __repr__(self):

        first_keys = (
            'diagnostic',
            'variable_group',
            'dataset',
            'project',
            'mip',
            'short_name',
        )

        def facets2str(facets):

            view = {k: facets[k] for k in first_keys if k in facets}
            for key, value in sorted(facets.items()):
                if key not in first_keys:
                    view[key] = value

            return pprint.pformat(view, sort_dicts=False)

        txt = [
            f"{self.__class__.__name__}:",
            facets2str(self.facets),
        ]
        if self.ancillaries:
            txt.append("ancillaries:")
            txt.extend(
                textwrap.indent(facets2str(a.facets), "  ")
                for a in self.ancillaries)
        return "\n".join(txt)

    def __getitem__(self, key):
        return self.facets[key]

    def __setitem__(self, key, value):
        self.facets[key] = value

    def set_facet(self, key, value, persist=True):
        self.facets[key] = value
        if persist:
            self._persist.add(key)

    @property
    def session(self):
        if self._session is None:
            raise ValueError(
                "Session not set, please create a session by using "
                "`esmvalcore.experimental.CFG.start_session` and "
                "and add it to this dataset.")
        return self._session

    @session.setter
    def session(self, session):
        self._session = session
        for ancillary in self.ancillaries:
            ancillary.session = session

    def add_ancillary(self, **facets):
        ancillary = self.copy(**facets)
        ancillary.ancillaries = []
        self.ancillaries.append(ancillary)

    def augment_facets(self, session=None):
        """Add extra facets."""
        if session is None:
            session = self.session
        self._augment_facets(session)
        for ancillary in self.ancillaries:
            ancillary._augment_facets(session)

    def _augment_facets(self, session):
        extra_facets = get_extra_facets(self, session['extra_facets_dir'])
        _augment(self.facets, extra_facets)
        if 'institute' not in self.facets:
            institute = get_institutes(self.facets)
            if institute:
                self.facets['institute'] = institute
        if 'activity' not in self.facets:
            activity = get_activity(self.facets)
            if activity:
                self.facets['activity'] = activity
        _get_facets_from_cmor_table(self.facets)
        _get_timerange_from_years(self.facets)
        if self.facets.get('frequency') == 'fx':
            self.facets.pop('timerange', None)

    def find_files(self, session: Session | None = None):
        """Find files."""
        if session is None:
            session = self.session
        self.augment_facets(session)
        self._find_files(session)
        for ancillary in self.ancillaries:
            ancillary._find_files(session)

    def _find_files(self, session):
        (input_files, dirnames,
         filenames) = get_input_filelist(self.facets,
                                         rootpath=session['rootpath'],
                                         drs=session['drs'])
        self.files = input_files
        self._files_debug = (dirnames, filenames)

        # Set up downloading from ESGF if requested.
        if (not session['offline']
                and self.facets['project'] in esgf.facets.FACETS):
            try:
                check_data_availability(self, log=False)
            except InputFilesNotFound:
                # Only look on ESGF if files are not available locally.
                local_files = set(Path(f).name for f in input_files)
                search_result = esgf.find_files(**self.facets)
                for file in search_result:
                    local_copy = file.local_file(session['download_dir'])
                    if local_copy.name not in local_files:
                        input_files.append(file)

                dirnames.append('ESGF:')

    @property
    def files(self):
        if self._files is None:
            self.find_files()
        return self._files

    @files.setter
    def files(self, value):
        self._files = value

    def expand(self) -> list['Dataset']:
        """Factory function that expands shorthands to generate datasets."""
        datasets = [self]
        for key in 'ensemble', 'sub_experiment':
            if key in self.facets:
                datasets = [
                    ds.copy(**{key: value}) for ds in datasets
                    for value in ds._expand_range(key)
                ]
        return datasets

    def _expand_range(self, input_tag):
        """Expand ranges such as ensemble members or stardates.

        Expansion only supports ensembles defined as strings, not lists.
        """
        expanded = []
        regex = re.compile(r'\(\d+:\d+\)')

        def expand_range(input_range):
            match = regex.search(input_range)
            if match:
                start, end = match.group(0)[1:-1].split(':')
                for i in range(int(start), int(end) + 1):
                    range_ = regex.sub(str(i), input_range, 1)
                    expand_range(range_)
            else:
                expanded.append(input_range)

        tag = self.facets.get(input_tag, "")
        if isinstance(tag, (list, tuple)):
            for elem in tag:
                if regex.search(elem):
                    raise RecipeError(
                        f"In {self}: {input_tag} expansion "
                        f"cannot be combined with {input_tag} lists")
            expanded.append(tag)
        else:
            expand_range(tag)

        return expanded

    def _update_timerange(self, session: Session | None = None):
        """Update wildcards in timerange with found datetime values.

        If the timerange is given as a year, it ensures it's formatted
        as a 4-digit value (YYYY).
        """
        if 'timerange' not in self.facets:
            return

        timerange = self.facets.pop('timerange')
        check_valid_time_selection(timerange)

        if '*' in timerange:
            self.find_files(session)
            if not self.files:
                raise InputFilesNotFound(
                    f"Missing data for: {_format_facets(self.facets)}. "
                    f"Cannot determine timerange '{timerange}'.")
            files = [
                f.name if isinstance(f, ESGFFile) else f for f in self.files
            ]
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
        check_valid_time_selection(timerange)

        self['timerange'] = timerange

    def load(self, session: Session | None = None) -> Cube:
        """Load dataset."""
        if session is None:
            session = self.session
        check_level = session['check_level']
        preproc_dir = session.preproc_dir
        cube = self._load(preproc_dir, check_level)
        fx_cubes = []
        for ancillary_dataset in self.ancillaries:
            if ancillary_dataset.files:
                fx_cube = ancillary_dataset._load(preproc_dir, check_level)
                fx_cubes.append(fx_cube)
        input_files = list(self.files)
        input_files.extend(anc.files for anc in self.ancillaries)
        cubes = preprocess(
            [cube],
            'add_fx_variables',
            input_files=input_files,
            fx_variables=fx_cubes,
        )
        return cubes[0]

    def _load(self, preproc_dir: Path, check_level) -> Cube:
        """Load self.files into an iris cube and return it."""
        output_file = get_output_file(self.facets, preproc_dir)

        settings = {}
        settings['fix_file'] = {
            'output_dir': Path(f"{output_file.with_suffix('')}_fixed"),
            **self.facets,
        }
        settings['load'] = {}
        settings['fix_metadata'] = {
            'check_level': check_level,
            **self.facets,
        }
        settings['concatenate'] = {}
        settings['cmor_check_metadata'] = {
            'check_level': check_level,
            'cmor_table': self.facets['project'],
            'mip': self.facets['mip'],
            'frequency': self.facets['frequency'],
            'short_name': self.facets['short_name'],
        }
        if 'timerange' in self.facets:
            settings['clip_timerange'] = {
                'timerange': self.facets['timerange'],
            }
        settings['fix_data'] = {
            'check_level': check_level,
            **self.facets,
        }
        settings['cmor_check_data'] = {
            'check_level': check_level,
            'cmor_table': self.facets['project'],
            'mip': self.facets['mip'],
            'frequency': self.facets['frequency'],
            'short_name': self.facets['short_name'],
        }

        result = self.files
        for step, kwargs in settings.items():
            result = preprocess(result, step, input_files=self.files, **kwargs)
        cube = result[0]
        return cube


def _augment(base, update):
    """Update dict base with values from dict update."""
    for key in update:
        if key not in base:
            base[key] = update[key]


ALIAS_INFO_KEYS = (
    'project',
    'activity',
    'dataset',
    'exp',
    'ensemble',
    'version',
)
"""List of keys to be used to compose the alias, ordered by priority."""


def _set_aliases(datasets):
    """Add a unique alias per diagnostic."""
    for _, group in groupby(datasets, key=lambda ds: ds.facets['diagnostic']):
        diag_datasets = [
            list(h)
            for _, h in groupby(group,
                                key=lambda ds: ds.facets['variable_group'])
        ]
        _set_alias(diag_datasets)


def _set_alias(variables):
    """Add unique alias for datasets.

    Generates a unique alias for each dataset that will be shared by all
    variables. Tries to make it as small as possible to make it useful for
    plot legends, filenames and such

    It is composed using the keys in Recipe.info_keys that differ from
    dataset to dataset. Once a diverging key is found, others are added
    to the alias only if the previous ones where not enough to fully
    identify the dataset.

    If key values are not strings, they will be joint using '-' if they
    are iterables or replaced by they string representation if they are not

    Function will not modify alias if it is manually added to the recipe
    but it will use the dataset info to compute the others

    Examples
    --------
    - {project: CMIP5, model: EC-Earth, ensemble: r1i1p1}
    - {project: CMIP6, model: EC-Earth, ensemble: r1i1p1f1}
    will generate alias 'CMIP5' and 'CMIP6'

    - {project: CMIP5, model: EC-Earth, experiment: historical}
    - {project: CMIP5, model: MPI-ESM, experiment: piControl}
    will generate alias 'EC-Earth,' and 'MPI-ESM'

    - {project: CMIP5, model: EC-Earth, experiment: historical}
    - {project: CMIP5, model: EC-Earth, experiment: piControl}
    will generate alias 'historical' and 'piControl'

    - {project: CMIP5, model: EC-Earth, experiment: historical}
    - {project: CMIP6, model: EC-Earth, experiment: historical}
    - {project: CMIP5, model: MPI-ESM, experiment: historical}
    - {project: CMIP6, model: MPI-ESM experiment: historical}
    will generate alias 'CMIP5_EC-EARTH', 'CMIP6_EC-EARTH', 'CMIP5_MPI-ESM'
    and 'CMIP6_MPI-ESM'

    - {project: CMIP5, model: EC-Earth, experiment: historical}
    will generate alias 'EC-Earth'

    Parameters
    ----------
    datasets : list
        for each variable, a list of datasets
    """
    datasets_info = set()

    def _key_str(obj):
        if isinstance(obj, str):
            return obj
        try:
            return '-'.join(obj)
        except TypeError:
            return str(obj)

    for variable in variables:
        for dataset in variable:
            alias = tuple(
                _key_str(dataset.facets.get(key, None))
                for key in ALIAS_INFO_KEYS)
            datasets_info.add(alias)
            if 'alias' not in dataset.facets:
                dataset.facets['alias'] = alias

    alias = dict()
    for info in datasets_info:
        alias[info] = []

    datasets_info = list(datasets_info)
    _get_next_alias(alias, datasets_info, 0)

    for info in datasets_info:
        alias[info] = '_'.join(
            [str(value) for value in alias[info] if value is not None])
        if not alias[info]:
            alias[info] = info[ALIAS_INFO_KEYS.index('dataset')]

    for variable in variables:
        for dataset in variable:
            dataset.facets['alias'] = alias.get(dataset.facets['alias'],
                                                dataset.facets['alias'])


def _get_next_alias(alias, datasets_info, i):
    if i >= len(ALIAS_INFO_KEYS):
        return
    key_values = set(info[i] for info in datasets_info)
    if len(key_values) == 1:
        for info in iter(datasets_info):
            alias[info].append(None)
    else:
        for info in datasets_info:
            alias[info].append(info[i])
    for key in key_values:
        _get_next_alias(
            alias,
            [info for info in datasets_info if info[i] == key],
            i + 1,
        )


def _merge_ancillary_dicts(var_facets, ds_facets):
    """Update the elements of `var_facets` with those in `ds_facets`.

    Both are lists of dicts containing facets
    """
    merged = {}
    msg = ("'short_name' is required for ancillary_variables entries, "
           "but missing in")
    for facets in var_facets:
        if 'short_name' not in facets:
            raise RecipeError(f"{msg} {facets}")
        else:
            merged[facets['short_name']] = facets
    for facets in ds_facets:
        if 'short_name' not in facets:
            raise RecipeError(f"{msg} {facets}")
        else:
            short_name = facets['short_name']
            if short_name not in merged:
                merged[short_name] = {}
            merged[short_name].update(facets)

    return list(merged.values())


def datasets_from_recipe(recipe, session):

    datasets = []

    for diagnostic in recipe['diagnostics'] or []:
        for variable_group in recipe['diagnostics'][diagnostic].get(
                'variables', {}):
            logger.debug(
                "Populating list of datasets for variable %s in diagnostic %s",
                variable_group, diagnostic)
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
            check_datasets(recipe_datasets, diagnostic, variable_group)

            idx = 0
            for recipe_dataset in recipe_datasets:
                DATASET_KEYS.union(recipe_dataset)
                recipe_dataset = copy.deepcopy(recipe_dataset)
                facets = copy.deepcopy(recipe_variable)
                facets.pop('additional_datasets', None)
                for key, value in recipe_dataset.items():
                    if key == 'ancillary_variables' and key in facets:
                        _merge_ancillary_dicts(facets[key], value)
                    else:
                        facets[key] = value

                persist = set(facets)
                facets['diagnostic'] = diagnostic
                facets['variable_group'] = variable_group
                if 'short_name' not in facets:
                    facets['short_name'] = variable_group
                    persist.add('short_name')
                check_variable(facets, required_keys=_REQUIRED_KEYS)
                preprocessor = str(facets.pop('preprocessor', 'default'))
                if facets['project'] == 'obs4mips':
                    logger.warning(
                        "Correcting capitalization, project 'obs4mips' "
                        "should be written as 'obs4MIPs'")
                    facets['project'] = 'obs4MIPs'
                if 'end_year' in facets and session['max_years']:
                    facets['end_year'] = min(
                        facets['end_year'],
                        facets['start_year'] + session['max_years'] - 1)
                ancillaries = facets.pop('ancillary_variables', [])
                dataset = Dataset()
                dataset.session = session
                for key, value in facets.items():
                    dataset.set_facet(key, value, key in persist)
                dataset.set_facet('preprocessor', preprocessor,
                                  preprocessor != 'default')
                for dataset in dataset.expand():
                    for ancillary_facets in ancillaries:
                        dataset.add_ancillary(**ancillary_facets)
                    dataset.facets['recipe_dataset_index'] = idx
                    datasets.append(dataset)
                    idx += 1

    _set_aliases(datasets)

    return datasets


def datasets_to_recipe(datasets):

    diagnostics = {}

    for dataset in datasets:
        if 'diagnostic' not in dataset.facets:
            raise RecipeError(
                f"'diagnostic' facet missing from dataset {dataset},"
                "unable to convert to recipe.")
        diagnostic = dataset.facets['diagnostic']
        if diagnostic not in diagnostics:
            diagnostics[diagnostic] = {'variables': {}}
        variables = diagnostics[diagnostic]['variables']
        if 'variable_group' in dataset.facets:
            variable_group = dataset.facets['variable_group']
        else:
            variable_group = dataset.facets['short_name']
        if variable_group not in variables:
            variables[variable_group] = {'additional_datasets': []}
        facets = dataset.minimal_facets
        if facets['short_name'] == variable_group:
            facets.pop('short_name')
        if dataset.ancillaries:
            facets['ancillary_variables'] = []
        for ancillary in dataset.ancillaries:
            anc_facets = {}
            for key, value in ancillary.minimal_facets.items():
                if facets.get(key) != value:
                    anc_facets[key] = value
            facets['ancillary_variables'].append(anc_facets)
        variables[variable_group]['additional_datasets'].append(facets)

    # TODO: make recipe look nice
    # - move identical facets from dataset to variable
    # - deduplicate by moving datasets up from variable to diagnostic to recipe
    # - remove variable_group if the same as short_name
    # - remove automatically added facets

    recipe = {'diagnostics': diagnostics}
    return recipe
