"""Recipe parser."""
import fnmatch
import logging
import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pformat

import yaml
from nested_lookup import get_all_keys, nested_delete, nested_lookup
from netCDF4 import Dataset

from . import __version__
from . import _recipe_checks as check
from . import esgf
from ._config import (
    TAGS,
    get_activity,
    get_extra_facets,
    get_institutes,
    get_project_config,
)
from ._data_finder import (
    _find_input_files,
    _get_timerange_from_years,
    _parse_period,
    _truncate_dates,
    dates_to_timerange,
    get_input_filelist,
    get_multiproduct_filename,
    get_output_file,
    get_start_end_date,
)
from ._provenance import TrackedFile, get_recipe_provenance
from ._task import DiagnosticTask, ResumeTask, TaskSet
from .cmor.check import CheckLevels
from .cmor.table import CMOR_TABLES
from .exceptions import InputFilesNotFound, RecipeError
from .preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
    PreprocessingTask,
    PreprocessorFile,
)
from .preprocessor._derive import get_required
from .preprocessor._io import DATASET_KEYS, concatenate_callback
from .preprocessor._other import _group_products
from .preprocessor._regrid import (
    _spec_to_latlonvals,
    get_cmor_levels,
    get_reference_levels,
    parse_cell_spec,
)

logger = logging.getLogger(__name__)

TASKSEP = os.sep

DOWNLOAD_FILES = set()
"""Use a global variable to keep track of files that need to be downloaded."""


def read_recipe_file(filename, config_user, initialize_tasks=True):
    """Read a recipe from file."""
    check.recipe_with_schema(filename)
    with open(filename, 'r') as file:
        raw_recipe = yaml.safe_load(file)

    return Recipe(raw_recipe,
                  config_user,
                  initialize_tasks,
                  recipe_file=filename)


def _add_cmor_info(variable, override=False):
    """Add information from CMOR tables to variable."""
    # Copy the following keys from CMOR table
    cmor_keys = [
        'standard_name', 'long_name', 'units', 'modeling_realm', 'frequency'
    ]
    project = variable['project']
    mip = variable['mip']
    short_name = variable['short_name']
    derive = variable.get('derive', False)
    table = CMOR_TABLES.get(project)
    if table:
        table_entry = table.get_variable(mip, short_name, derive)
    else:
        table_entry = None
    if table_entry is None:
        raise RecipeError(
            f"Unable to load CMOR table (project) '{project}' for variable "
            f"'{short_name}' with mip '{mip}'")
    variable['original_short_name'] = table_entry.short_name
    for key in cmor_keys:
        if key not in variable or override:
            value = getattr(table_entry, key, None)
            if value is not None:
                variable[key] = value
            else:
                logger.debug(
                    "Failed to add key %s to variable %s from CMOR table", key,
                    variable)

    # Check that keys are available
    check.variable(variable, required_keys=cmor_keys)


def _add_extra_facets(variable, extra_facets_dir):
    """Add extra_facets to variable."""
    extra_facets = get_extra_facets(variable["project"], variable["dataset"],
                                    variable["mip"], variable["short_name"],
                                    extra_facets_dir)
    _augment(variable, extra_facets)


def _special_name_to_dataset(variable, special_name):
    """Convert special names to dataset names."""
    if special_name in ('reference_dataset', 'alternative_dataset'):
        if special_name not in variable:
            raise RecipeError(
                "Preprocessor {preproc} uses {name}, but {name} is not "
                "defined for variable {short_name} of diagnostic "
                "{diagnostic}".format(
                    preproc=variable['preprocessor'],
                    name=special_name,
                    short_name=variable['short_name'],
                    diagnostic=variable['diagnostic'],
                ))
        special_name = variable[special_name]

    return special_name


def _update_target_levels(variable, variables, settings, config_user):
    """Replace the target levels dataset name with a filename if needed."""
    levels = settings.get('extract_levels', {}).get('levels')
    if not levels:
        return

    levels = _special_name_to_dataset(variable, levels)

    # If levels is a dataset name, replace it by a dict with a 'dataset' entry
    if any(levels == v['dataset'] for v in variables):
        settings['extract_levels']['levels'] = {'dataset': levels}
        levels = settings['extract_levels']['levels']

    if not isinstance(levels, dict):
        return

    if 'cmor_table' in levels and 'coordinate' in levels:
        settings['extract_levels']['levels'] = get_cmor_levels(
            levels['cmor_table'], levels['coordinate'])
    elif 'dataset' in levels:
        dataset = levels['dataset']
        if variable['dataset'] == dataset:
            del settings['extract_levels']
        else:
            variable_data = _get_dataset_info(dataset, variables)
            filename = _dataset_to_file(variable_data, config_user)
            fix_dir = f"{os.path.splitext(variable_data['filename'])[0]}_fixed"
            settings['extract_levels']['levels'] = get_reference_levels(
                filename=filename,
                project=variable_data['project'],
                dataset=dataset,
                short_name=variable_data['short_name'],
                mip=variable_data['mip'],
                frequency=variable_data['frequency'],
                fix_dir=fix_dir,
            )


def _update_target_grid(variable, variables, settings, config_user):
    """Replace the target grid dataset name with a filename if needed."""
    grid = settings.get('regrid', {}).get('target_grid')
    if not grid:
        return

    grid = _special_name_to_dataset(variable, grid)

    if variable['dataset'] == grid:
        del settings['regrid']
    elif any(grid == v['dataset'] for v in variables):
        settings['regrid']['target_grid'] = _dataset_to_file(
            _get_dataset_info(grid, variables), config_user)
    else:
        # Check that MxN grid spec is correct
        target_grid = settings['regrid']['target_grid']
        if isinstance(target_grid, str):
            parse_cell_spec(target_grid)
        # Check that cdo spec is correct
        elif isinstance(target_grid, dict):
            _spec_to_latlonvals(**target_grid)


def _update_regrid_time(variable, settings):
    """Input data frequency automatically for regrid_time preprocessor."""
    regrid_time = settings.get('regrid_time')
    if regrid_time is None:
        return
    frequency = settings.get('regrid_time', {}).get('frequency')
    if not frequency:
        settings['regrid_time']['frequency'] = variable['frequency']


def _get_dataset_info(dataset, variables):
    for var in variables:
        if var['dataset'] == dataset:
            return var
    raise RecipeError("Unable to find matching file for dataset"
                      "{}".format(dataset))


def _augment(base, update):
    """Update dict base with values from dict update."""
    for key in update:
        if key not in base:
            base[key] = update[key]


def _dataset_to_file(variable, config_user):
    """Find the first file belonging to dataset from variable info."""
    (files, dirnames, filenames) = _get_input_files(variable, config_user)
    if not files and variable.get('derive'):
        required_vars = get_required(variable['short_name'],
                                     variable['project'])
        for required_var in required_vars:
            _augment(required_var, variable)
            _add_cmor_info(required_var, override=True)
            _add_extra_facets(required_var, config_user['extra_facets_dir'])
            (files, dirnames,
             filenames) = _get_input_files(required_var, config_user)
            if files:
                variable = required_var
                break
    check.data_availability(files, variable, dirnames, filenames)
    return files[0]


def _limit_datasets(variables, profile, max_datasets=0):
    """Try to limit the number of datasets to max_datasets."""
    if not max_datasets:
        return variables

    logger.info("Limiting the number of datasets to %s", max_datasets)

    required_datasets = [
        (profile.get('extract_levels') or {}).get('levels'),
        (profile.get('regrid') or {}).get('target_grid'),
        variables[0].get('reference_dataset'),
        variables[0].get('alternative_dataset'),
    ]

    limited = [v for v in variables if v['dataset'] in required_datasets]
    for variable in variables:
        if len(limited) >= max_datasets:
            break
        if variable not in limited:
            limited.append(variable)

    logger.info("Only considering %s", ', '.join(v['alias'] for v in limited))

    return limited


def _get_default_settings(variable, config_user, derive=False):
    """Get default preprocessor settings."""
    settings = {}

    # Configure loading
    settings['load'] = {
        'callback': concatenate_callback,
    }
    # Configure concatenation
    settings['concatenate'] = {}

    # Configure fixes
    fix = deepcopy(variable)
    # File fixes
    fix_dir = os.path.splitext(variable['filename'])[0] + '_fixed'
    settings['fix_file'] = dict(fix)
    settings['fix_file']['output_dir'] = fix_dir
    # Cube fixes
    fix['frequency'] = variable['frequency']
    fix['check_level'] = config_user.get('check_level', CheckLevels.DEFAULT)
    settings['fix_metadata'] = dict(fix)
    settings['fix_data'] = dict(fix)

    # Configure time extraction
    if 'timerange' in variable and variable['frequency'] != 'fx':
        settings['clip_timerange'] = {'timerange': variable['timerange']}

    if derive:
        settings['derive'] = {
            'short_name': variable['short_name'],
            'standard_name': variable['standard_name'],
            'long_name': variable['long_name'],
            'units': variable['units'],
        }

    # Configure CMOR metadata check
    settings['cmor_check_metadata'] = {
        'cmor_table': variable['project'],
        'mip': variable['mip'],
        'short_name': variable['short_name'],
        'frequency': variable['frequency'],
        'check_level': config_user.get('check_level', CheckLevels.DEFAULT)
    }
    # Configure final CMOR data check
    settings['cmor_check_data'] = dict(settings['cmor_check_metadata'])

    # Clean up fixed files
    if not config_user['save_intermediary_cubes']:
        settings['cleanup'] = {
            'remove': [fix_dir],
        }

    # Configure saving cubes to file
    settings['save'] = {'compress': config_user['compress_netcdf']}
    if variable['short_name'] != variable['original_short_name']:
        settings['save']['alias'] = variable['short_name']

    # Configure fx settings
    settings['add_fx_variables'] = {
        'fx_variables': {},
        'check_level': config_user.get('check_level', CheckLevels.DEFAULT)
    }
    settings['remove_fx_variables'] = {}

    return settings


def _add_fxvar_keys(fx_info, variable, extra_facets_dir):
    """Add keys specific to fx variable to use get_input_filelist."""
    fx_variable = deepcopy(variable)
    fx_variable.update(fx_info)
    fx_variable['variable_group'] = fx_info['short_name']

    # add special ensemble for CMIP5 only
    if fx_variable['project'] == 'CMIP5':
        fx_variable['ensemble'] = 'r0i0p0'

    # add missing cmor info
    _add_cmor_info(fx_variable, override=True)

    # add extra_facets
    _add_extra_facets(fx_variable, extra_facets_dir)

    return fx_variable


def _search_fx_mip(tables, variable, fx_info, config_user):
    """Search mip for fx variable."""
    # Get all mips that offer that specific fx variable
    mips_with_fx_var = []
    for (mip, table) in tables.items():
        if fx_info['short_name'] in table:
            mips_with_fx_var.append(mip)

    # List is empty -> no table includes the fx variable
    if not mips_with_fx_var:
        raise RecipeError(
            f"Requested fx variable '{fx_info['short_name']}' not available "
            f"in any CMOR table for '{variable['project']}'")

    # Iterate through all possible mips and check if files are available; in
    # case of ambiguity raise an error
    fx_files_for_mips = {}
    for mip in mips_with_fx_var:
        fx_info['mip'] = mip
        fx_info = _add_fxvar_keys(fx_info, variable,
                                  config_user['extra_facets_dir'])
        logger.debug("For fx variable '%s', found table '%s'",
                     fx_info['short_name'], mip)
        fx_files = _get_input_files(fx_info, config_user)[0]
        if fx_files:
            logger.debug("Found fx variables '%s':\n%s", fx_info['short_name'],
                         pformat(fx_files))
            fx_files_for_mips[mip] = fx_files

    # Dict contains more than one element -> ambiguity
    if len(fx_files_for_mips) > 1:
        raise RecipeError(
            f"Requested fx variable '{fx_info['short_name']}' for dataset "
            f"'{variable['dataset']}' of project '{variable['project']}' is "
            f"available in more than one CMOR table for "
            f"'{variable['project']}': {sorted(list(fx_files_for_mips))}")

    # Dict is empty -> no files found -> handled at later stage
    if not fx_files_for_mips:
        fx_info['mip'] = variable['mip']
        fx_files = []

    # Dict contains one element -> ok
    else:
        mip = list(fx_files_for_mips)[0]
        fx_info['mip'] = mip
        fx_info = _add_fxvar_keys(fx_info, variable,
                                  config_user['extra_facets_dir'])
        fx_files = fx_files_for_mips[mip]

    return fx_info, fx_files


def _get_fx_files(variable, fx_info, config_user):
    """Get fx files (searching all possible mips)."""
    # assemble info from master variable
    var_project = variable['project']
    # check if project in config-developer
    try:
        get_project_config(var_project)
    except ValueError:
        raise RecipeError(f"Requested fx variable '{fx_info['short_name']}' "
                          f"with parent variable '{variable}' does not have "
                          f"a '{var_project}' project in config-developer.")
    project_tables = CMOR_TABLES[var_project].tables

    # If mip is not given, search all available tables. If the variable is not
    # found or files are available in more than one table, raise error
    if not fx_info['mip']:
        fx_info, fx_files = _search_fx_mip(project_tables, variable, fx_info,
                                           config_user)
    else:
        mip = fx_info['mip']
        if mip not in project_tables:
            raise RecipeError(
                f"Requested mip table '{mip}' for fx variable "
                f"'{fx_info['short_name']}' not available for project "
                f"'{var_project}'")
        if fx_info['short_name'] not in project_tables[mip]:
            raise RecipeError(
                f"fx variable '{fx_info['short_name']}' not available in CMOR "
                f"table '{mip}' for '{var_project}'")
        fx_info = _add_fxvar_keys(fx_info, variable,
                                  config_user['extra_facets_dir'])
        fx_files = _get_input_files(fx_info, config_user)[0]

    # Flag a warning if no files are found
    if not fx_files:
        logger.warning("Missing data for fx variable '%s' of dataset %s",
                       fx_info['short_name'],
                       fx_info['alias'].replace('_', ' '))

    # If frequency = fx, only allow a single file
    if fx_files:
        if fx_info['frequency'] == 'fx':
            fx_files = fx_files[0]

    return fx_files, fx_info


def _exclude_dataset(settings, variable, step):
    """Exclude dataset from specific preprocessor step if requested."""
    exclude = {
        _special_name_to_dataset(variable, dataset)
        for dataset in settings[step].pop('exclude', [])
    }
    if variable['dataset'] in exclude:
        settings.pop(step)
        logger.debug("Excluded dataset '%s' from preprocessor step '%s'",
                     variable['dataset'], step)


def _update_weighting_settings(settings, variable):
    """Update settings for the weighting preprocessors."""
    if 'weighting_landsea_fraction' not in settings:
        return
    _exclude_dataset(settings, variable, 'weighting_landsea_fraction')


def _update_fx_files(step_name, settings, variable, config_user, fx_vars):
    """Update settings with mask fx file list or dict."""
    if not fx_vars:
        return
    for fx_var, fx_info in fx_vars.items():
        if not fx_info:
            fx_info = {}
        if 'mip' not in fx_info:
            fx_info.update({'mip': None})
        if 'short_name' not in fx_info:
            fx_info.update({'short_name': fx_var})
        fx_files, fx_info = _get_fx_files(variable, fx_info, config_user)
        if fx_files:
            fx_info['filename'] = fx_files
            settings['add_fx_variables']['fx_variables'].update(
                {fx_var: fx_info})
            logger.debug('Using fx files for variable %s during step %s: %s',
                         variable['short_name'], step_name, pformat(fx_files))


def _fx_list_to_dict(fx_vars):
    """Convert fx list to dictionary.

    To be deprecated at some point.
    """
    user_fx_vars = {}
    for fx_var in fx_vars:
        if isinstance(fx_var, dict):
            short_name = fx_var['short_name']
            user_fx_vars.update({short_name: fx_var})
            continue
        user_fx_vars.update({fx_var: None})
    return user_fx_vars


def _update_fx_settings(settings, variable, config_user):
    """Update fx settings depending on the needed method."""
    # Add default values to the option 'fx_variables' if it is not explicitly
    # specified and transform fx variables to dicts
    def _update_fx_vars_in_settings(step_settings, step_name):
        """Update fx_variables option in the settings."""
        # Add default values for fx_variables
        if 'fx_variables' not in step_settings:
            default_fx = {
                'area_statistics': {
                    'areacella': None,
                },
                'mask_landsea': {
                    'sftlf': None,
                },
                'mask_landseaice': {
                    'sftgif': None,
                },
                'volume_statistics': {
                    'volcello': None,
                },
                'weighting_landsea_fraction': {
                    'sftlf': None,
                },
            }
            if variable['project'] != 'obs4MIPs':
                default_fx['area_statistics']['areacello'] = None
                default_fx['mask_landsea']['sftof'] = None
                default_fx['weighting_landsea_fraction']['sftof'] = None
            step_settings['fx_variables'] = default_fx[step_name]

        # Transform fx variables to dicts
        user_fx_vars = step_settings['fx_variables']
        if user_fx_vars is None:
            step_settings['fx_variables'] = {}
        elif isinstance(user_fx_vars, list):
            step_settings['fx_variables'] = _fx_list_to_dict(user_fx_vars)

    fx_steps = [
        'mask_landsea', 'mask_landseaice', 'weighting_landsea_fraction',
        'area_statistics', 'volume_statistics'
    ]
    for step_name in settings:
        if step_name in fx_steps:
            _update_fx_vars_in_settings(settings[step_name], step_name)
            _update_fx_files(step_name, settings, variable, config_user,
                             settings[step_name]['fx_variables'])
            # Remove unused attribute in 'fx_steps' preprocessors.
            # The fx_variables information is saved in
            # the 'add_fx_variables' step.
            settings[step_name].pop('fx_variables', None)


def _read_attributes(filename):
    """Read the attributes from a netcdf file."""
    attributes = {}
    if not (os.path.exists(filename)
            and os.path.splitext(filename)[1].lower() == '.nc'):
        return attributes

    with Dataset(filename, 'r') as dataset:
        for attr in dataset.ncattrs():
            attributes[attr] = dataset.getncattr(attr)
    return attributes


def _get_input_files(variable, config_user):
    """Get the input files for a single dataset (locally and via download)."""
    if variable['frequency'] != 'fx':
        start_year, end_year = _parse_period(variable['timerange'])

        start_year = int(str(start_year[0:4]))
        end_year = int(str(end_year[0:4]))

        variable['start_year'] = start_year
        variable['end_year'] = end_year
    (input_files, dirnames,
     filenames) = get_input_filelist(variable=variable,
                                     rootpath=config_user['rootpath'],
                                     drs=config_user['drs'])

    # Set up downloading from ESGF if requested.
    if (not config_user['offline']
            and variable['project'] in esgf.facets.FACETS):
        try:
            check.data_availability(
                input_files,
                variable,
                dirnames,
                filenames,
                log=False,
            )
        except RecipeError:
            # Only look on ESGF if files are not available locally.
            local_files = set(Path(f).name for f in input_files)
            search_result = esgf.find_files(**variable)
            for file in search_result:
                local_copy = file.local_file(config_user['download_dir'])
                if local_copy.name not in local_files:
                    if not local_copy.exists():
                        DOWNLOAD_FILES.add(file)
                    input_files.append(str(local_copy))

            dirnames.append('ESGF:')

    return (input_files, dirnames, filenames)


def _get_ancestors(variable, config_user):
    """Get the input files for a single dataset and setup provenance."""
    (input_files, dirnames,
     filenames) = _get_input_files(variable, config_user)

    logger.debug(
        "Using input files for variable %s of dataset %s:\n%s",
        variable['short_name'],
        variable['alias'].replace('_', ' '),
        '\n'.join(
            f'{f} (will be downloaded)' if not os.path.exists(f) else str(f)
            for f in input_files),
    )
    check.data_availability(input_files, variable, dirnames, filenames)
    logger.info("Found input files for %s",
                variable['alias'].replace('_', ' '))

    # Set up provenance tracking
    for i, filename in enumerate(input_files):
        attributes = _read_attributes(filename)
        input_files[i] = TrackedFile(filename, attributes)

    return input_files


def _apply_preprocessor_profile(settings, profile_settings):
    """Apply settings from preprocessor profile."""
    profile_settings = deepcopy(profile_settings)
    for step, args in profile_settings.items():
        # Remove disabled preprocessor functions
        if args is False:
            if step in settings:
                del settings[step]
            continue
        # Enable/update functions without keywords
        if step not in settings:
            settings[step] = {}
        if isinstance(args, dict):
            settings[step].update(args)


def _get_common_attributes(products, settings):
    """Get common attributes for the output products."""
    attributes = {}
    some_product = next(iter(products))
    for key, value in some_product.attributes.items():
        if all(p.attributes.get(key, object()) == value for p in products):
            attributes[key] = value

    # Ensure that attribute timerange is always available. This depends on the
    # "span" setting: if "span=overlap", the intersection of all periods is
    # used; if "span=full", the union is used. The default value for "span" is
    # "overlap".
    span = settings.get('span', 'overlap')
    for product in products:
        timerange = product.attributes['timerange']
        start, end = _parse_period(timerange)
        if 'timerange' not in attributes:
            attributes['timerange'] = dates_to_timerange(start, end)
        else:
            start_date, end_date = _parse_period(attributes['timerange'])
            start_date, start = _truncate_dates(start_date, start)
            end_date, end = _truncate_dates(end_date, end)

            # If "span=overlap", always use the latest start_date and the
            # earliest end_date
            if span == 'overlap':
                start_date = max([start, start_date])
                end_date = min([end, end_date])

            # If "span=full", always use the earliest start_date and the latest
            # end_date. Note: span can only take the values "overlap" or "full"
            # (this is checked earlier).
            else:
                start_date = min([start, start_date])
                end_date = max([end, end_date])

            attributes['timerange'] = dates_to_timerange(start_date, end_date)

    # Ensure that attributes start_year and end_year are always available
    start_year, end_year = _parse_period(attributes['timerange'])
    attributes['start_year'] = int(str(start_year[0:4]))
    attributes['end_year'] = int(str(end_year[0:4]))

    return attributes


def _get_downstream_settings(step, order, products):
    """Get downstream preprocessor settings shared between products."""
    settings = {}
    remaining_steps = order[order.index(step) + 1:]
    some_product = next(iter(products))
    for key, value in some_product.settings.items():
        if key in remaining_steps:
            if all(p.settings.get(key, object()) == value for p in products):
                settings[key] = value
    return settings


def _update_multi_dataset_settings(variable, settings):
    """Configure multi dataset statistics."""
    for step in MULTI_MODEL_FUNCTIONS:
        if not settings.get(step):
            continue
        # Exclude dataset if requested
        _exclude_dataset(settings, variable, step)


def _update_warning_settings(settings, project):
    """Update project-specific warning settings."""
    cfg = get_project_config(project)
    if 'ignore_warnings' not in cfg:
        return
    for (step, ignored_warnings) in cfg['ignore_warnings'].items():
        if step in settings:
            settings[step]['ignore_warnings'] = ignored_warnings


def _get_tag(step, identifier, statistic):
    # Avoid . in filename for percentiles
    statistic = statistic.replace('.', '-')

    if step == 'ensemble_statistics':
        tag = 'Ensemble' + statistic.title()
    elif identifier == '':
        tag = 'MultiModel' + statistic.title()
    else:
        tag = identifier + statistic.title()

    return tag


def _update_multiproduct(input_products, order, preproc_dir, step):
    """Return new products that are aggregated over multiple datasets.

    These new products will replace the original products at runtime.
    Therefore, they need to have all the settings for the remaining steps.

    The functions in _multimodel.py take output_products as function arguments.
    These are the output_products created here. But since those functions are
    called from the input products, the products that are created here need to
    be added to their ancestors products' settings ().
    """
    products = {p for p in input_products if step in p.settings}
    if not products:
        return input_products, {}

    settings = list(products)[0].settings[step]

    if step == 'ensemble_statistics':
        check.ensemble_statistics_preproc(settings)
        grouping = ['project', 'dataset', 'exp', 'sub_experiment']
    else:
        check.multimodel_statistics_preproc(settings)
        grouping = settings.get('groupby', None)

    downstream_settings = _get_downstream_settings(step, order, products)

    relevant_settings = {
        'output_products': defaultdict(dict)
    }  # pass to ancestors

    output_products = set()
    for identifier, products in _group_products(products, by_key=grouping):
        common_attributes = _get_common_attributes(products, settings)

        for statistic in settings.get('statistics', []):
            statistic_attributes = dict(common_attributes)
            statistic_attributes[step] = _get_tag(step, identifier, statistic)
            statistic_attributes.setdefault('alias',
                                            statistic_attributes[step])
            statistic_attributes.setdefault('dataset',
                                            statistic_attributes[step])
            filename = get_multiproduct_filename(statistic_attributes,
                                                 preproc_dir)
            statistic_attributes['filename'] = filename
            statistic_product = PreprocessorFile(statistic_attributes,
                                                 downstream_settings)
            output_products.add(statistic_product)
            relevant_settings['output_products'][identifier][
                statistic] = statistic_product

    return output_products, relevant_settings


def update_ancestors(ancestors, step, downstream_settings):
    """Retroactively add settings to ancestor products."""
    for product in ancestors:
        if step in product.settings:
            settings = product.settings[step]
            for key, value in downstream_settings.items():
                settings[key] = value


def _update_extract_shape(settings, config_user):
    if 'extract_shape' in settings:
        shapefile = settings['extract_shape'].get('shapefile')
        if shapefile:
            if not os.path.exists(shapefile):
                shapefile = os.path.join(
                    config_user['auxiliary_data_dir'],
                    shapefile,
                )
                settings['extract_shape']['shapefile'] = shapefile
        check.extract_shape(settings['extract_shape'])


def _update_timerange(variable, config_user):
    """Update wildcards in timerange with found datetime values.

    If the timerange is given as a year, it ensures it's formatted as a
    4-digit value (YYYY).
    """
    if 'timerange' not in variable:
        return

    timerange = variable.get('timerange')
    check.valid_time_selection(timerange)

    if '*' in timerange:
        (files, _, _) = _find_input_files(
            variable, config_user['rootpath'], config_user['drs'])
        if not files and not config_user.get('offline', True):
            facets = deepcopy(variable)
            facets.pop('timerange', None)
            files = [file.name for file in esgf.find_files(**facets)]

        if not files:
            raise InputFilesNotFound(
                f"Missing data for {variable['alias']}: "
                f"{variable['short_name']}. Cannot determine indeterminate "
                f"time range '{timerange}'."
            )

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

    variable['timerange'] = timerange


def _match_products(products, variables):
    """Match a list of input products to output product attributes."""
    grouped_products = defaultdict(list)

    if not products:
        return grouped_products

    def get_matching(attributes):
        """Find the output filename which matches input attributes best."""
        best_score = 0
        filenames = []
        for variable in variables:
            filename = variable['filename']
            score = sum(v == variable.get(k) for k, v in attributes.items())

            if score > best_score:
                best_score = score
                filenames = [filename]
            elif score == best_score:
                filenames.append(filename)

        if not filenames:
            logger.warning(
                "Unable to find matching output file for input file %s",
                filename)

        return filenames

    # Group input files by output file
    for product in products:
        matching_filenames = get_matching(product.attributes)
        for filename in matching_filenames:
            grouped_products[filename].append(product)

    return grouped_products


def _allow_skipping(ancestors, variable, config_user):
    """Allow skipping of datasets."""
    allow_skipping = all([
        config_user.get('skip_nonexistent'),
        not ancestors,
        variable['dataset'] != variable.get('reference_dataset'),
    ])
    return allow_skipping


def _get_preprocessor_products(variables, profile, order, ancestor_products,
                               config_user, name):
    """Get preprocessor product definitions for a set of datasets.

    It updates recipe settings as needed by various preprocessors and
    sets the correct ancestry.
    """
    products = set()
    preproc_dir = config_user['preproc_dir']

    for variable in variables:
        _update_timerange(variable, config_user)
        variable['filename'] = get_output_file(variable,
                                               config_user['preproc_dir'])

    if ancestor_products:
        grouped_ancestors = _match_products(ancestor_products, variables)
    else:
        grouped_ancestors = {}

    missing_vars = set()
    for variable in variables:
        settings = _get_default_settings(
            variable,
            config_user,
            derive='derive' in profile,
        )
        _update_warning_settings(settings, variable['project'])
        _apply_preprocessor_profile(settings, profile)
        _update_multi_dataset_settings(variable, settings)
        try:
            _update_target_levels(
                variable=variable,
                variables=variables,
                settings=settings,
                config_user=config_user,
            )
        except RecipeError as ex:
            missing_vars.add(ex.message)
        _update_preproc_functions(settings, config_user, variable, variables,
                                  missing_vars)
        ancestors = grouped_ancestors.get(variable['filename'])
        if not ancestors:
            try:
                ancestors = _get_ancestors(variable, config_user)
            except RecipeError as ex:
                if _allow_skipping(ancestors, variable, config_user):
                    logger.info("Skipping: %s", ex.message)
                else:
                    missing_vars.add(ex.message)
                continue
        product = PreprocessorFile(
            attributes=variable,
            settings=settings,
            ancestors=ancestors,
        )

        products.add(product)

    if missing_vars:
        separator = "\n- "
        raise InputFilesNotFound(
            f'Missing data for preprocessor {name}:{separator}'
            f'{separator.join(sorted(missing_vars))}')

    check.reference_for_bias_preproc(products)

    ensemble_step = 'ensemble_statistics'
    multi_model_step = 'multi_model_statistics'
    if ensemble_step in profile:
        ensemble_products, ensemble_settings = _update_multiproduct(
            products, order, preproc_dir, ensemble_step)

        # check for ensemble_settings to bypass tests
        update_ancestors(
            ancestors=products,
            step=ensemble_step,
            downstream_settings=ensemble_settings,
        )
    else:
        ensemble_products = products

    if multi_model_step in profile:
        multimodel_products, multimodel_settings = _update_multiproduct(
            ensemble_products, order, preproc_dir, multi_model_step)

        # check for multi_model_settings to bypass tests
        update_ancestors(
            ancestors=products,
            step=multi_model_step,
            downstream_settings=multimodel_settings,
        )

        if ensemble_step in profile:
            # Update multi-product settings (workaround for lack of better
            # ancestry tracking)
            update_ancestors(
                ancestors=ensemble_products,
                step=multi_model_step,
                downstream_settings=multimodel_settings,
            )
    else:
        multimodel_products = set()

    for product in products | multimodel_products | ensemble_products:
        product.check()

        # Ensure that attributes start_year and end_year are always available
        # for all products if a timerange is specified
        if 'timerange' in product.attributes:
            start_year, end_year = _parse_period(
                product.attributes['timerange'])
            product.attributes['start_year'] = int(str(start_year[0:4]))
            product.attributes['end_year'] = int(str(end_year[0:4]))

    return products


def _update_preproc_functions(settings, config_user, variable, variables,
                              missing_vars):
    _update_extract_shape(settings, config_user)
    _update_weighting_settings(settings, variable)
    _update_fx_settings(settings=settings,
                        variable=variable,
                        config_user=config_user)
    _update_timerange(variable, config_user)
    try:
        _update_target_grid(
            variable=variable,
            variables=variables,
            settings=settings,
            config_user=config_user,
        )
    except RecipeError as ex:
        missing_vars.add(ex.message)
    _update_regrid_time(variable, settings)


def _get_single_preprocessor_task(variables,
                                  profile,
                                  config_user,
                                  name,
                                  ancestor_tasks=None):
    """Create preprocessor tasks for a set of datasets w/ special case fx."""
    if ancestor_tasks is None:
        ancestor_tasks = []
    order = _extract_preprocessor_order(profile)
    ancestor_products = [p for task in ancestor_tasks for p in task.products]

    if variables[0].get('frequency') == 'fx':
        check.check_for_temporal_preprocs(profile)
        ancestor_products = None

    products = _get_preprocessor_products(
        variables=variables,
        profile=profile,
        order=order,
        ancestor_products=ancestor_products,
        config_user=config_user,
        name=name,
    )

    if not products:
        raise RecipeError(
            "Did not find any input data for task {}".format(name))

    task = PreprocessingTask(
        products=products,
        ancestors=ancestor_tasks,
        name=name,
        order=order,
        debug=config_user['save_intermediary_cubes'],
        write_ncl_interface=config_user['write_ncl_interface'],
    )

    logger.info("PreprocessingTask %s created.", task.name)
    logger.debug("PreprocessingTask %s will create the files:\n%s", task.name,
                 '\n'.join(p.filename for p in task.products))

    return task


def _extract_preprocessor_order(profile):
    """Extract the order of the preprocessing steps from the profile."""
    custom_order = profile.pop('custom_order', False)
    if not custom_order:
        return DEFAULT_ORDER
    order = tuple(p for p in profile if p not in INITIAL_STEPS + FINAL_STEPS)
    return INITIAL_STEPS + order + FINAL_STEPS


def _split_settings(settings, step, order=DEFAULT_ORDER):
    """Split settings, using step as a separator."""
    before = {}
    for _step in order:
        if _step == step:
            break
        if _step in settings:
            before[_step] = settings[_step]
    after = {
        k: v
        for k, v in settings.items() if not (k == step or k in before)
    }
    return before, after


def _split_derive_profile(profile):
    """Split the derive preprocessor profile."""
    order = _extract_preprocessor_order(profile)
    before, after = _split_settings(profile, 'derive', order)
    after['derive'] = True
    after['fix_file'] = False
    after['fix_metadata'] = False
    after['fix_data'] = False
    if order != DEFAULT_ORDER:
        before['custom_order'] = True
        after['custom_order'] = True
    return before, after


def _check_differing_timeranges(timeranges, required_vars):
    """Log error if required variables have differing timeranges."""
    if len(timeranges) > 1:
        raise ValueError(
            f"Differing timeranges with values {timeranges} "
            f"found for required variables {required_vars}. "
            "Set `timerange` to a common value.",
        )


def _get_derive_input_variables(variables, config_user):
    """Determine the input sets of `variables` needed for deriving."""
    derive_input = {}

    def append(group_prefix, var):
        """Append variable `var` to a derive input group."""
        group = group_prefix + var['short_name']
        var['variable_group'] = group
        if group not in derive_input:
            derive_input[group] = []
        derive_input[group].append(var)

    for variable in variables:
        group_prefix = variable['variable_group'] + '_derive_input_'
        if not variable.get('force_derivation') and \
           '*' in variable['timerange']:
            raise RecipeError(
                f"Error in derived variable: {variable['short_name']}: "
                "Using 'force_derivation: false' (the default option) "
                "in combination with wildcards ('*') in timerange is "
                "not allowed; explicitly use 'force_derivation: true' "
                "or avoid the use of wildcards in timerange."
                )
        if not variable.get('force_derivation') and _get_input_files(
           variable, config_user)[0]:
            # No need to derive, just process normally up to derive step
            var = deepcopy(variable)
            append(group_prefix, var)
        else:
            # Process input data needed to derive variable
            required_vars = get_required(variable['short_name'],
                                         variable['project'])
            timeranges = set()
            for var in required_vars:
                _augment(var, variable)
                _add_cmor_info(var, override=True)
                _add_extra_facets(var, config_user['extra_facets_dir'])
                _update_timerange(var, config_user)
                files = _get_input_files(var, config_user)[0]
                if var.get('optional') and not files:
                    logger.info(
                        "Skipping: no data found for %s which is marked as "
                        "'optional'", var)
                else:
                    append(group_prefix, var)
                    timeranges.add(var['timerange'])
            _check_differing_timeranges(timeranges, required_vars)
            variable['timerange'] = " ".join(timeranges)

    # An empty derive_input (due to all variables marked as 'optional' is
    # handled at a later step
    return derive_input


def _get_preprocessor_task(variables, profiles, config_user, task_name):
    """Create preprocessor task(s) for a set of datasets."""
    # First set up the preprocessor profile
    variable = variables[0]
    preproc_name = variable.get('preprocessor')
    if preproc_name not in profiles:
        raise RecipeError(
            "Unknown preprocessor {} in variable {} of diagnostic {}".format(
                preproc_name, variable['short_name'], variable['diagnostic']))
    profile = deepcopy(profiles[variable['preprocessor']])
    logger.info("Creating preprocessor '%s' task for variable '%s'",
                variable['preprocessor'], variable['short_name'])
    variables = _limit_datasets(variables, profile,
                                config_user.get('max_datasets'))
    for variable in variables:
        _add_cmor_info(variable)
    # Create preprocessor task(s)
    derive_tasks = []
    # set up tasks
    if variable.get('derive'):
        # Create tasks to prepare the input data for the derive step
        derive_profile, profile = _split_derive_profile(profile)
        derive_input = _get_derive_input_variables(variables, config_user)

        for derive_variables in derive_input.values():
            for derive_variable in derive_variables:
                _add_cmor_info(derive_variable, override=True)
            derive_name = task_name.split(
                TASKSEP)[0] + TASKSEP + derive_variables[0]['variable_group']
            task = _get_single_preprocessor_task(
                derive_variables,
                derive_profile,
                config_user,
                name=derive_name,
            )
            derive_tasks.append(task)

    # Create (final) preprocessor task
    task = _get_single_preprocessor_task(
        variables,
        profile,
        config_user,
        ancestor_tasks=derive_tasks,
        name=task_name,
    )

    return task


class Recipe:
    """Recipe object."""

    info_keys = ('project', 'activity', 'dataset', 'exp', 'ensemble',
                 'version')
    """List of keys to be used to compose the alias, ordered by priority."""

    def __init__(self,
                 raw_recipe,
                 config_user,
                 initialize_tasks=True,
                 recipe_file=None):
        """Parse a recipe file into an object."""
        # Clear the global variable containing the set of files to download
        DOWNLOAD_FILES.clear()
        self._download_files = set()
        self._cfg = deepcopy(config_user)
        self._cfg['write_ncl_interface'] = self._need_ncl(
            raw_recipe['diagnostics'])
        self._raw_recipe = raw_recipe
        self._updated_recipe = {}
        self._filename = os.path.basename(recipe_file)
        self._preprocessors = raw_recipe.get('preprocessors', {})
        if 'default' not in self._preprocessors:
            self._preprocessors['default'] = {}
        self.diagnostics = self._initialize_diagnostics(
            raw_recipe['diagnostics'], raw_recipe.get('datasets', []))
        self.entity = self._initialize_provenance(
            raw_recipe.get('documentation', {}))
        try:
            self.tasks = self.initialize_tasks() if initialize_tasks else None
        except RecipeError as exc:
            self._log_recipe_errors(exc)
            raise

    def _log_recipe_errors(self, exc):
        """Log a message with recipe errors."""
        logger.error(exc.message)
        for task in exc.failed_tasks:
            logger.error(task.message)

        if self._cfg['offline'] and any(
                isinstance(err, InputFilesNotFound)
                for err in exc.failed_tasks):
            logger.error(
                "Not all input files required to run the recipe could be"
                " found.")
            logger.error(
                "If the files are available locally, please check"
                " your `rootpath` and `drs` settings in your user "
                "configuration file %s", self._cfg['config_file'])
            logger.error(
                "To automatically download the required files to "
                "`download_dir: %s`, set `offline: false` in %s or run the "
                "recipe with the extra command line argument --offline=False",
                self._cfg['download_dir'],
                self._cfg['config_file'],
            )
            logger.info(
                "Note that automatic download is only available for files"
                " that are hosted on the ESGF, i.e. for projects: %s, and %s",
                ', '.join(list(esgf.facets.FACETS)[:-1]),
                list(esgf.facets.FACETS)[-1],
            )

    @staticmethod
    def _need_ncl(raw_diagnostics):
        if not raw_diagnostics:
            return False
        for diagnostic in raw_diagnostics.values():
            if not diagnostic.get('scripts'):
                continue
            for script in diagnostic['scripts'].values():
                if script.get('script', '').lower().endswith('.ncl'):
                    logger.info("NCL script detected, checking NCL version")
                    check.ncl_version()
                    return True
        return False

    def _initialize_provenance(self, raw_documentation):
        """Initialize the recipe provenance."""
        doc = deepcopy(raw_documentation)

        TAGS.replace_tags_in_dict(doc)

        return get_recipe_provenance(doc, self._filename)

    def _initialize_diagnostics(self, raw_diagnostics, raw_datasets):
        """Define diagnostics in recipe."""
        logger.debug("Retrieving diagnostics from recipe")
        check.diagnostics(raw_diagnostics)

        diagnostics = {}

        for name, raw_diagnostic in raw_diagnostics.items():
            diagnostic = {}
            diagnostic['name'] = name
            additional_datasets = raw_diagnostic.get('additional_datasets', [])
            datasets = (raw_datasets + additional_datasets)
            diagnostic['preprocessor_output'] = \
                self._initialize_preprocessor_output(
                    name,
                    raw_diagnostic.get('variables', {}),
                    datasets)
            variable_names = tuple(raw_diagnostic.get('variables', {}))
            diagnostic['scripts'] = self._initialize_scripts(
                name, raw_diagnostic.get('scripts'), variable_names)
            for key in ('themes', 'realms'):
                if key in raw_diagnostic:
                    for script in diagnostic['scripts'].values():
                        script['settings'][key] = raw_diagnostic[key]
            diagnostics[name] = diagnostic

        return diagnostics

    @staticmethod
    def _initialize_datasets(raw_datasets):
        """Define datasets used by variable."""
        datasets = deepcopy(raw_datasets)

        for dataset in datasets:
            for key in dataset:
                DATASET_KEYS.add(key)
        return datasets

    @staticmethod
    def _expand_tag(variables, input_tag):
        """Expand tags such as ensemble members or startdates.

        Expansion only supports ensembles defined as strings, not lists.
        Returns the expanded datasets.
        """
        expanded = []
        regex = re.compile(r'\(\d+:\d+\)')

        def expand_tag(variable, input_tag):
            tag = variable.get(input_tag, "")
            match = regex.search(tag)
            if match:
                start, end = match.group(0)[1:-1].split(':')
                for i in range(int(start), int(end) + 1):
                    expand = deepcopy(variable)
                    expand[input_tag] = regex.sub(str(i), tag, 1)
                    expand_tag(expand, input_tag)
            else:
                expanded.append(variable)

        for variable in variables:
            tag = variable.get(input_tag, "")
            if isinstance(tag, (list, tuple)):
                for elem in tag:
                    if regex.search(elem):
                        raise RecipeError(
                            f"In variable {variable}: {input_tag} expansion "
                            f"cannot be combined with {input_tag} lists")
                expanded.append(variable)
            else:
                expand_tag(variable, input_tag)

        return expanded

    def _initialize_variables(self, raw_variable, raw_datasets):
        """Define variables for all datasets."""
        variables = []

        raw_variable = deepcopy(raw_variable)
        datasets = self._initialize_datasets(
            raw_datasets + raw_variable.pop('additional_datasets', []))
        if not datasets:
            raise RecipeError("You have not specified any dataset "
                              "or additional_dataset groups "
                              f"for variable {raw_variable} Exiting.")
        check.duplicate_datasets(datasets)

        for index, dataset in enumerate(datasets):
            variable = deepcopy(raw_variable)
            variable.update(dataset)

            variable['recipe_dataset_index'] = index
            if 'end_year' in variable and self._cfg.get('max_years'):
                variable['end_year'] = min(
                    variable['end_year'],
                    variable['start_year'] + self._cfg['max_years'] - 1)
            variables.append(variable)

        required_keys = {
            'short_name',
            'mip',
            'dataset',
            'project',
            'preprocessor',
            'diagnostic',
        }
        if 'fx' not in raw_variable.get('mip', ''):
            required_keys.update({'timerange'})
        else:
            variable.pop('timerange', None)
        for variable in variables:
            _add_extra_facets(variable, self._cfg['extra_facets_dir'])
            _get_timerange_from_years(variable)
            if 'institute' not in variable:
                institute = get_institutes(variable)
                if institute:
                    variable['institute'] = institute
            if 'activity' not in variable:
                activity = get_activity(variable)
                if activity:
                    variable['activity'] = activity
            if 'sub_experiment' in variable:
                subexperiment_keys = deepcopy(required_keys)
                subexperiment_keys.update({'sub_experiment'})
                check.variable(variable, subexperiment_keys)
            else:
                check.variable(variable, required_keys)
            if variable['project'] == 'obs4mips':
                logger.warning("Correcting capitalization, project 'obs4mips'"
                               " should be written as 'obs4MIPs'")
                variable['project'] = 'obs4MIPs'
        variables = self._expand_tag(variables, 'ensemble')
        variables = self._expand_tag(variables, 'sub_experiment')

        return variables

    def _initialize_preprocessor_output(self, diagnostic_name, raw_variables,
                                        raw_datasets):
        """Define variables in diagnostic."""
        logger.debug("Populating list of variables for diagnostic %s",
                     diagnostic_name)

        preprocessor_output = {}

        for variable_group, raw_variable in raw_variables.items():
            if raw_variable is None:
                raw_variable = {}
            else:
                raw_variable = deepcopy(raw_variable)
            raw_variable['variable_group'] = variable_group
            if 'short_name' not in raw_variable:
                raw_variable['short_name'] = variable_group
            raw_variable['diagnostic'] = diagnostic_name
            raw_variable['preprocessor'] = str(
                raw_variable.get('preprocessor', 'default'))
            preprocessor_output[variable_group] = \
                self._initialize_variables(raw_variable, raw_datasets)

        self._set_alias(preprocessor_output)

        return preprocessor_output

    def _set_alias(self, preprocessor_output):
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
        preprocessor_output : dict
            preprocessor output dictionary
        """
        datasets_info = set()

        def _key_str(obj):
            if isinstance(obj, str):
                return obj
            try:
                return '-'.join(obj)
            except TypeError:
                return str(obj)

        for variable in preprocessor_output.values():
            for dataset in variable:
                alias = tuple(
                    _key_str(dataset.get(key, None)) for key in self.info_keys)
                datasets_info.add(alias)
                if 'alias' not in dataset:
                    dataset['alias'] = alias

        alias = dict()
        for info in datasets_info:
            alias[info] = []

        datasets_info = list(datasets_info)
        self._get_next_alias(alias, datasets_info, 0)

        for info in datasets_info:
            alias[info] = '_'.join(
                [str(value) for value in alias[info] if value is not None])
            if not alias[info]:
                alias[info] = info[self.info_keys.index('dataset')]

        for variable in preprocessor_output.values():
            for dataset in variable:
                dataset['alias'] = alias.get(dataset['alias'],
                                             dataset['alias'])

    @classmethod
    def _get_next_alias(cls, alias, datasets_info, i):
        if i >= len(cls.info_keys):
            return
        key_values = set(info[i] for info in datasets_info)
        if len(key_values) == 1:
            for info in iter(datasets_info):
                alias[info].append(None)
        else:
            for info in datasets_info:
                alias[info].append(info[i])
        for key in key_values:
            cls._get_next_alias(
                alias,
                [info for info in datasets_info if info[i] == key],
                i + 1,
            )

    def _initialize_scripts(self, diagnostic_name, raw_scripts,
                            variable_names):
        """Define script in diagnostic."""
        if not raw_scripts:
            return {}

        logger.debug("Setting script for diagnostic %s", diagnostic_name)

        scripts = {}

        for script_name, raw_settings in raw_scripts.items():
            settings = deepcopy(raw_settings)
            script = settings.pop('script')
            ancestors = []
            for id_glob in settings.pop('ancestors', variable_names):
                if TASKSEP not in id_glob:
                    id_glob = diagnostic_name + TASKSEP + id_glob
                ancestors.append(id_glob)
            settings['recipe'] = self._filename
            settings['version'] = __version__
            settings['script'] = script_name
            # Add output dirs to settings
            for dir_name in ('run_dir', 'plot_dir', 'work_dir'):
                settings[dir_name] = os.path.join(self._cfg[dir_name],
                                                  diagnostic_name, script_name)
            # Copy other settings
            if self._cfg['write_ncl_interface']:
                settings['exit_on_ncl_warning'] = self._cfg['exit_on_warning']
            for key in (
                    'output_file_type',
                    'log_level',
                    'profile_diagnostic',
                    'auxiliary_data_dir',
            ):
                settings[key] = self._cfg[key]

            scripts[script_name] = {
                'script': script,
                'output_dir': settings['work_dir'],
                'settings': settings,
                'ancestors': ancestors,
            }

        return scripts

    def _resolve_diagnostic_ancestors(self, tasks):
        """Resolve diagnostic ancestors."""
        tasks = {t.name: t for t in tasks}
        for diagnostic_name, diagnostic in self.diagnostics.items():
            for script_name, script_cfg in diagnostic['scripts'].items():
                task_id = diagnostic_name + TASKSEP + script_name
                if task_id in tasks and isinstance(tasks[task_id],
                                                   DiagnosticTask):
                    logger.debug("Linking tasks for diagnostic %s script %s",
                                 diagnostic_name, script_name)
                    ancestors = []
                    for id_glob in script_cfg['ancestors']:
                        ancestor_ids = fnmatch.filter(tasks, id_glob)
                        if not ancestor_ids:
                            raise RecipeError(
                                "Could not find any ancestors matching {}".
                                format(id_glob))
                        logger.debug("Pattern %s matches %s", id_glob,
                                     ancestor_ids)
                        ancestors.extend(tasks[a] for a in ancestor_ids)
                    tasks[task_id].ancestors = ancestors

    def _get_tasks_to_run(self):
        """Get tasks filtered and add ancestors if needed."""
        tasknames_to_run = self._cfg.get('diagnostics', [])
        if tasknames_to_run:
            tasknames_to_run = set(tasknames_to_run)
            while self._update_with_ancestors(tasknames_to_run):
                pass
        return tasknames_to_run

    def _update_with_ancestors(self, tasknames_to_run):
        """Add ancestors for all selected tasks."""
        num_filters = len(tasknames_to_run)

        # Iterate over all tasks and add all ancestors to tasknames_to_run of
        # those tasks that match one of the patterns given by tasknames_to_run
        # to
        for diagnostic_name, diagnostic in self.diagnostics.items():
            for script_name, script_cfg in diagnostic['scripts'].items():
                task_name = diagnostic_name + TASKSEP + script_name
                for pattern in tasknames_to_run:
                    if fnmatch.fnmatch(task_name, pattern):
                        ancestors = script_cfg.get('ancestors', [])
                        if isinstance(ancestors, str):
                            ancestors = ancestors.split()
                        for ancestor in ancestors:
                            tasknames_to_run.add(ancestor)
                        break

        # If new ancestors have been added (num_filters !=
        # len(tasknames_to_run)) -> return True. This causes another call of
        # this function in the while() loop of _get_tasks_to_run to ensure that
        # nested ancestors are found.

        # If no new ancestors have been found (num_filters ==
        # len(tasknames_to_run)) -> return False. This terminates the search
        # for ancestors.

        return num_filters != len(tasknames_to_run)

    def _create_diagnostic_tasks(self, diagnostic_name, diagnostic,
                                 tasknames_to_run):
        """Create diagnostic tasks."""
        tasks = []

        if self._cfg.get('run_diagnostic', True):
            for script_name, script_cfg in diagnostic['scripts'].items():
                task_name = diagnostic_name + TASKSEP + script_name

                # Skip diagnostic tasks if desired by the user
                if tasknames_to_run:
                    for pattern in tasknames_to_run:
                        if fnmatch.fnmatch(task_name, pattern):
                            break
                    else:
                        logger.info("Skipping task %s due to filter",
                                    task_name)
                        continue

                logger.info("Creating diagnostic task %s", task_name)
                task = DiagnosticTask(
                    script=script_cfg['script'],
                    output_dir=script_cfg['output_dir'],
                    settings=script_cfg['settings'],
                    name=task_name,
                )
                tasks.append(task)

        return tasks

    def _fill_wildcards(self, variable_group, preprocessor_output):
        """Fill wildcards in the `timerange` .

        The new values will be datetime values that have been found for
        the first and/or last available points.
        """
        # To be generalised for other tags
        datasets = self._raw_recipe.get('datasets')
        diagnostics = self._raw_recipe.get('diagnostics')
        additional_datasets = []
        if diagnostics:
            additional_datasets = nested_lookup('additional_datasets',
                                                diagnostics)

        raw_dataset_tags = nested_lookup('timerange', datasets)
        raw_diagnostic_tags = nested_lookup('timerange', diagnostics)

        wildcard = False
        for raw_timerange in raw_dataset_tags + raw_diagnostic_tags:
            if '*' in raw_timerange:
                wildcard = True
                break

        if wildcard:
            if not self._updated_recipe:
                self._updated_recipe = deepcopy(self._raw_recipe)
                nested_delete(self._updated_recipe, 'datasets', in_place=True)
                nested_delete(self._updated_recipe,
                              'additional_datasets',
                              in_place=True)
            updated_datasets = []
            dataset_keys = set(
                get_all_keys(datasets) + get_all_keys(additional_datasets) +
                ['timerange'])
            for data in preprocessor_output[variable_group]:
                diagnostic = data['diagnostic']
                updated_datasets.append(
                    {key: data[key]
                     for key in dataset_keys if key in data})
            self._updated_recipe['diagnostics'][diagnostic]['variables'][
                variable_group].pop('timerange', None)
            self._updated_recipe['diagnostics'][diagnostic]['variables'][
                variable_group].update(
                    {'additional_datasets': updated_datasets})

    def _create_preprocessor_tasks(self, diagnostic_name, diagnostic,
                                   tasknames_to_run, any_diag_script_is_run):
        """Create preprocessor tasks."""
        tasks = []
        failed_tasks = []
        for variable_group in diagnostic['preprocessor_output']:
            task_name = diagnostic_name + TASKSEP + variable_group

            # Skip preprocessor if not a single diagnostic script is run and
            # the preprocessing task is not explicitly requested by the user
            if tasknames_to_run:
                if not any_diag_script_is_run:
                    for pattern in tasknames_to_run:
                        if fnmatch.fnmatch(task_name, pattern):
                            break
                    else:
                        logger.info("Skipping task %s due to filter",
                                    task_name)
                        continue

            # Resume previous runs if requested, else create a new task
            for resume_dir in self._cfg['resume_from']:
                prev_preproc_dir = Path(
                    resume_dir,
                    'preproc',
                    diagnostic_name,
                    variable_group,
                )
                if prev_preproc_dir.exists():
                    logger.info("Re-using preprocessed files from %s for %s",
                                prev_preproc_dir, task_name)
                    preproc_dir = Path(
                        self._cfg['preproc_dir'],
                        'preproc',
                        diagnostic_name,
                        variable_group,
                    )
                    task = ResumeTask(prev_preproc_dir, preproc_dir, task_name)
                    tasks.append(task)
                    break
            else:
                logger.info("Creating preprocessor task %s", task_name)
                try:
                    task = _get_preprocessor_task(
                        variables=diagnostic['preprocessor_output']
                        [variable_group],
                        profiles=self._preprocessors,
                        config_user=self._cfg,
                        task_name=task_name,
                    )
                except RecipeError as ex:
                    failed_tasks.append(ex)
                else:
                    self._fill_wildcards(variable_group,
                                         diagnostic['preprocessor_output'])
                    tasks.append(task)

        return tasks, failed_tasks

    def _create_tasks(self):
        """Create tasks from the recipe."""
        logger.info("Creating tasks from recipe")
        tasks = TaskSet()

        tasknames_to_run = self._get_tasks_to_run()

        priority = 0
        failed_tasks = []

        for diagnostic_name, diagnostic in self.diagnostics.items():
            logger.info("Creating tasks for diagnostic %s", diagnostic_name)

            # Create diagnostic tasks
            new_tasks = self._create_diagnostic_tasks(diagnostic_name,
                                                      diagnostic,
                                                      tasknames_to_run)
            any_diag_script_is_run = bool(new_tasks)
            for task in new_tasks:
                task.priority = priority
                tasks.add(task)
                priority += 1

            # Create preprocessor tasks
            new_tasks, failed = self._create_preprocessor_tasks(
                diagnostic_name, diagnostic, tasknames_to_run,
                any_diag_script_is_run)
            failed_tasks.extend(failed)
            for task in new_tasks:
                for task0 in task.flatten():
                    task0.priority = priority
                tasks.add(task)
                priority += 1

        if failed_tasks:
            recipe_error = RecipeError('Could not create all tasks')
            recipe_error.failed_tasks.extend(failed_tasks)
            raise recipe_error

        check.tasks_valid(tasks)

        # Resolve diagnostic ancestors
        if self._cfg.get('run_diagnostic', True):
            self._resolve_diagnostic_ancestors(tasks)

        return tasks

    def initialize_tasks(self):
        """Define tasks in recipe."""
        tasks = self._create_tasks()
        tasks = tasks.flatten()
        logger.info("These tasks will be executed: %s",
                    ', '.join(t.name for t in tasks))

        # Initialize task provenance
        for task in tasks:
            task.initialize_provenance(self.entity)

        # Store the set of files to download before running
        self._download_files = set(DOWNLOAD_FILES)

        # Return smallest possible set of tasks
        return tasks.get_independent()

    def __str__(self):
        """Get human readable summary."""
        return '\n\n'.join(str(task) for task in self.tasks)

    def run(self):
        """Run all tasks in the recipe."""
        self.write_filled_recipe()
        if not self.tasks:
            raise RecipeError('No tasks to run!')

        # Download required data
        if not self._cfg['offline']:
            esgf.download(self._download_files, self._cfg['download_dir'])

        self.tasks.run(max_parallel_tasks=self._cfg['max_parallel_tasks'])
        self.write_html_summary()

    def get_output(self) -> dict:
        """Return the paths to the output plots and data.

        Returns
        -------
        product_filenames : dict
            Lists of products/attributes grouped by task.
        """
        output = {}

        output['recipe_config'] = self._cfg
        output['recipe_filename'] = self._filename
        output['recipe_data'] = self._raw_recipe
        output['task_output'] = {}

        for task in self.tasks.flatten():
            if self._cfg['remove_preproc_dir'] and isinstance(
                    task, PreprocessingTask):
                # Skip preprocessing tasks that are deleted afterwards
                continue
            output['task_output'][task.name] = task.get_product_attributes()

        return output

    def write_filled_recipe(self):
        """Write copy of recipe with filled wildcards."""
        if self._updated_recipe:
            run_dir = self._cfg['run_dir']
            filename = self._filename.split('.')
            filename[0] = filename[0] + '_filled'
            new_filename = '.'.join(filename)
            with open(os.path.join(run_dir, new_filename), 'w') as file:
                yaml.safe_dump(self._updated_recipe, file, sort_keys=False)

    def write_html_summary(self):
        """Write summary html file to the output dir."""
        with warnings.catch_warnings():
            # ignore import warnings
            warnings.simplefilter("ignore")
            # keep RecipeOutput here to avoid circular import
            from esmvalcore.experimental.recipe_output import RecipeOutput
            output = self.get_output()

            try:
                output = RecipeOutput.from_core_recipe_output(output)
            except LookupError as error:
                # See https://github.com/ESMValGroup/ESMValCore/issues/28
                logger.warning("Could not write HTML report: %s", error)
            else:
                output.write_html()
