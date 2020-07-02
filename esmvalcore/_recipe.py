"""Recipe parser."""
import fnmatch
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from pprint import pformat

import yaml
from netCDF4 import Dataset

from . import __version__
from . import _recipe_checks as check
from ._config import (TAGS, get_activity, get_institutes, get_project_config,
                      replace_tags)
from ._data_finder import (get_input_filelist, get_output_file,
                           get_statistic_output_file)
from ._provenance import TrackedFile, get_recipe_provenance
from ._recipe_checks import RecipeError
from ._task import (DiagnosticTask, get_flattened_tasks, get_independent_tasks,
                    run_tasks)
from .cmor.table import CMOR_TABLES
from .preprocessor import (DEFAULT_ORDER, FINAL_STEPS, INITIAL_STEPS,
                           MULTI_MODEL_FUNCTIONS, PreprocessingTask,
                           PreprocessorFile)
from .preprocessor._derive import get_required
from .preprocessor._download import synda_search
from .preprocessor._io import DATASET_KEYS, concatenate_callback
from .preprocessor._regrid import (get_cmor_levels, get_reference_levels,
                                   parse_cell_spec)

logger = logging.getLogger(__name__)

TASKSEP = os.sep


def ordered_safe_load(stream):
    """Load a YAML file using OrderedDict instead of dict."""
    class OrderedSafeLoader(yaml.SafeLoader):
        """Loader class that uses OrderedDict to load a map."""
    def construct_mapping(loader, node):
        """Load a map as an OrderedDict."""
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return yaml.load(stream, OrderedSafeLoader)


def load_raw_recipe(filename):
    """Check a recipe file and return it in raw form."""
    # Note that many checks can only be performed after the automatically
    # computed entries have been filled in by creating a Recipe object.
    check.recipe_with_schema(filename)
    with open(filename, 'r') as file:
        contents = file.read()
        raw_recipe = yaml.safe_load(contents)
        raw_recipe['preprocessors'] = ordered_safe_load(contents).get(
            'preprocessors', {})

    check.diagnostics(raw_recipe['diagnostics'])
    return raw_recipe


def read_recipe_file(filename, config_user, initialize_tasks=True):
    """Read a recipe from file."""
    raw_recipe = load_raw_recipe(filename)
    return Recipe(raw_recipe,
                  config_user,
                  initialize_tasks,
                  recipe_file=filename)


def _get_value(key, datasets):
    """Get a value for key by looking at the other datasets."""
    values = {dataset[key] for dataset in datasets if key in dataset}

    if len(values) > 1:
        raise RecipeError("Ambiguous values {} for property {}".format(
            values, key))

    value = None
    if len(values) == 1:
        value = values.pop()

    return value


def _add_cmor_info(variable, override=False):
    """Add information from CMOR tables to variable."""
    logger.debug("If not present: adding keys from CMOR table to %s", variable)
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
            settings['extract_levels']['levels'] = get_reference_levels(
                filename=filename,
                project=variable_data['project'],
                dataset=dataset,
                short_name=variable_data['short_name'],
                mip=variable_data['mip'],
                frequency=variable_data['frequency'],
                fix_dir=os.path.splitext(variable_data['filename'])[0] +
                '_fixed',
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
        parse_cell_spec(settings['regrid']['target_grid'])


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

    # Set up downloading using synda if requested.
    if config_user['synda_download']:
        # TODO: make this respect drs or download to preproc dir?
        download_folder = os.path.join(config_user['preproc_dir'], 'downloads')
        settings['download'] = {
            'dest_folder': download_folder,
        }

    # Configure loading
    settings['load'] = {
        'callback': concatenate_callback,
    }
    # Configure concatenation
    settings['concatenate'] = {}

    # Configure fixes
    fix = {
        'project': variable['project'],
        'dataset': variable['dataset'],
        'short_name': variable['short_name'],
        'mip': variable['mip'],
    }
    # File fixes
    fix_dir = os.path.splitext(variable['filename'])[0] + '_fixed'
    settings['fix_file'] = dict(fix)
    settings['fix_file']['output_dir'] = fix_dir
    # Cube fixes
    fix['frequency'] = variable['frequency']
    fix['check_level'] = config_user['check_level']
    settings['fix_metadata'] = dict(fix)
    settings['fix_data'] = dict(fix)

    # Configure time extraction
    if 'start_year' in variable and 'end_year' in variable \
            and variable['frequency'] != 'fx':
        settings['extract_time'] = {
            'start_year': variable['start_year'],
            'end_year': variable['end_year'] + 1,
            'start_month': 1,
            'end_month': 1,
            'start_day': 1,
            'end_day': 1,
        }

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
        'check_level': config_user['check_level']
    }
    # Configure final CMOR data check
    settings['cmor_check_data'] = {
        'cmor_table': variable['project'],
        'mip': variable['mip'],
        'short_name': variable['short_name'],
        'frequency': variable['frequency'],
        'check_level': config_user['check_level']
    }

    # Clean up fixed files
    if not config_user['save_intermediary_cubes']:
        settings['cleanup'] = {
            'remove': [fix_dir],
        }

    # Configure saving cubes to file
    settings['save'] = {'compress': config_user['compress_netcdf']}

    return settings


def _add_fxvar_keys(fx_var_dict, variable):
    """Add keys specific to fx variable to use get_input_filelist."""
    fx_variable = dict(variable)
    fx_variable.update(fx_var_dict)

    # set variable names
    fx_variable['variable_group'] = fx_var_dict['short_name']

    # add special ensemble for CMIP5 only
    if fx_variable['project'] == 'CMIP5':
        fx_variable['ensemble'] = 'r0i0p0'

    # add missing cmor info
    _add_cmor_info(fx_variable, override=True)

    return fx_variable


def _get_fx_file(variable, fx_variable, config_user):
    """Get fx files (searching all possible mips)."""
    # make it a dict
    if isinstance(fx_variable, str):
        fx_varname = fx_variable
        fx_variable = {'short_name': fx_varname}
    else:
        fx_varname = fx_variable['short_name']

    # assemble info from master variable
    var = dict(variable)
    var_project = variable['project']
    # check if project in config-developer
    try:
        get_project_config(var_project)
    except ValueError:
        raise RecipeError(
            f"Requested fx variable '{fx_varname}' with parent variable"
            f"'{variable}' does not have a '{var_project}' project"
            f"in config-developer.")
    cmor_table = CMOR_TABLES[var_project]
    valid_fx_vars = []

    # force only the mip declared by user
    if 'mip' in fx_variable:
        fx_mips = [fx_variable['mip']]
    else:
        # Get all fx-related mips (original var mip,
        # 'fx' and extend from cmor tables)
        fx_mips = [variable['mip']]
        fx_mips.extend(mip for mip in cmor_table.tables if 'fx' in mip)

    # Search all mips for available variables
    # priority goes to user specified mip if available
    searched_mips = []
    fx_files = []
    for fx_mip in fx_mips:
        fx_cmor_variable = cmor_table.get_variable(fx_mip, fx_varname)
        if fx_cmor_variable is not None:
            fx_var_dict = dict(fx_variable)
            searched_mips.append(fx_mip)
            fx_var_dict['mip'] = fx_mip
            fx_var_dict = _add_fxvar_keys(fx_var_dict, var)
            valid_fx_vars.append(fx_var_dict)
            logger.debug("For fx variable '%s', found table '%s'", fx_varname,
                         fx_mip)
            fx_files = _get_input_files(fx_var_dict, config_user)[0]

            # If files found, return them
            if fx_files:
                logger.debug("Found fx variables '%s':\n%s", fx_varname,
                             pformat(fx_files))
                break

    # If fx variable was not found in any table, raise exception
    if not searched_mips:
        raise RecipeError(
            f"Requested fx variable '{fx_varname}' not available in "
            f"any 'fx'-related CMOR table ({fx_mips}) for '{var_project}'")

    # flag a warning
    if not fx_files:
        logger.warning("Missing data for fx variable '%s'", fx_varname)

    # allow for empty lists corrected for by NE masks
    if fx_files:
        fx_files = fx_files[0]
    if valid_fx_vars:
        valid_fx_vars = valid_fx_vars[0]

    return fx_files, valid_fx_vars


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

    fx_vars = [
        _get_fx_file(variable, fxvar, config_user)
        for fxvar in fx_vars
    ]

    fx_dict = {fx_var[1]['short_name']: fx_var[0] for fx_var in fx_vars}
    settings['fx_variables'] = fx_dict
    logger.info('Using fx_files: %s for variable %s during step %s',
                pformat(settings['fx_variables']),
                variable['short_name'],
                step_name)


def _update_fx_settings(settings, variable, config_user):
    """Update fx settings depending on the needed method."""
    # get fx variables either from user defined attribute or fixed
    def _get_fx_vars_from_attribute(step_settings, step_name):
        user_fx_vars = step_settings.get('fx_variables')
        if not user_fx_vars:
            if step_name in ('mask_landsea', 'weighting_landsea_fraction'):
                user_fx_vars = ['sftlf']
                if variable['project'] != 'obs4mips':
                    user_fx_vars.append('sftof')
            elif step_name == 'mask_landseaice':
                user_fx_vars = ['sftgif']
            elif step_name in ('area_statistics',
                               'volume_statistics', 'zonal_statistics'):
                user_fx_vars = []
        return user_fx_vars

    fx_steps = [
        'mask_landsea', 'mask_landseaice', 'weighting_landsea_fraction',
        'area_statistics', 'volume_statistics', 'zonal_statistics'
    ]

    for step_name, step_settings in settings.items():
        if step_name in fx_steps:
            fx_vars = _get_fx_vars_from_attribute(step_settings, step_name)
            _update_fx_files(step_name, step_settings,
                             variable, config_user, fx_vars)


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
    (input_files, dirnames,
     filenames) = get_input_filelist(variable=variable,
                                     rootpath=config_user['rootpath'],
                                     drs=config_user['drs'])

    # Set up downloading using synda if requested.
    # Do not download if files are already available locally.
    if config_user['synda_download'] and not input_files:
        input_files = synda_search(variable)
        dirnames = None
        filenames = None

    return (input_files, dirnames, filenames)


def _get_ancestors(variable, config_user):
    """Get the input files for a single dataset and setup provenance."""
    (input_files, dirnames,
     filenames) = _get_input_files(variable, config_user)

    logger.info("Using input files for variable %s of dataset %s:\n%s",
                variable['short_name'], variable['dataset'],
                '\n'.join(input_files))
    if (not config_user.get('skip-nonexistent')
            or variable['dataset'] == variable.get('reference_dataset')):
        check.data_availability(input_files, variable, dirnames, filenames)

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


def _get_statistic_attributes(products):
    """Get attributes for the statistic output products."""
    attributes = {}
    some_product = next(iter(products))
    for key, value in some_product.attributes.items():
        if all(p.attributes.get(key, object()) == value for p in products):
            attributes[key] = value

    # Ensure start_year and end_year attributes are available
    for product in products:
        start = product.attributes['start_year']
        if 'start_year' not in attributes or start < attributes['start_year']:
            attributes['start_year'] = start
        end = product.attributes['end_year']
        if 'end_year' not in attributes or end > attributes['end_year']:
            attributes['end_year'] = end

    return attributes


def _get_remaining_common_settings(step, order, products):
    """Get preprocessor settings that are shared between products."""
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


def _update_statistic_settings(products, order, preproc_dir):
    """Define statistic output products."""
    # TODO: move this to multi model statistics function?
    # But how to check, with a dry-run option?
    step = 'multi_model_statistics'

    products = {p for p in products if step in p.settings}
    if not products:
        return

    some_product = next(iter(products))
    for statistic in some_product.settings[step]['statistics']:
        check.valid_multimodel_statistic(statistic)
        attributes = _get_statistic_attributes(products)
        attributes['dataset'] = attributes['alias'] = 'MultiModel{}'.format(
            statistic.title().replace('.', '-'))
        attributes['filename'] = get_statistic_output_file(
            attributes, preproc_dir)
        common_settings = _get_remaining_common_settings(step, order, products)
        statistic_product = PreprocessorFile(attributes, common_settings)
        for product in products:
            settings = product.settings[step]
            if 'output_products' not in settings:
                settings['output_products'] = {}
            settings['output_products'][statistic] = statistic_product


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


def _match_products(products, variables):
    """Match a list of input products to output product attributes."""
    grouped_products = {}

    def get_matching(attributes):
        """Find the output filename which matches input attributes best."""
        score = 0
        filenames = []
        for variable in variables:
            filename = variable['filename']
            tmp = sum(v == variable.get(k) for k, v in attributes.items())
            if tmp > score:
                score = tmp
                filenames = [filename]
            elif tmp == score:
                filenames.append(filename)
        if not filenames:
            logger.warning(
                "Unable to find matching output file for input file %s",
                filename)
        return filenames

    # Group input files by output file
    for product in products:
        for filename in get_matching(product.attributes):
            if filename not in grouped_products:
                grouped_products[filename] = []
            grouped_products[filename].append(product)

    return grouped_products


def _get_preprocessor_products(variables,
                               profile,
                               order,
                               ancestor_products,
                               config_user):
    """
    Get preprocessor product definitions for a set of datasets.

    It updates recipe settings as needed by various preprocessors
    and sets the correct ancestry.
    """
    products = set()
    for variable in variables:
        variable['filename'] = get_output_file(variable,
                                               config_user['preproc_dir'])

    if ancestor_products:
        grouped_ancestors = _match_products(ancestor_products, variables)
    else:
        grouped_ancestors = {}

    for variable in variables:
        settings = _get_default_settings(
            variable,
            config_user,
            derive='derive' in profile,
        )
        _apply_preprocessor_profile(settings, profile)
        _update_multi_dataset_settings(variable, settings)
        _update_target_levels(
            variable=variable,
            variables=variables,
            settings=settings,
            config_user=config_user,
        )
        _update_extract_shape(settings, config_user)
        _update_weighting_settings(settings, variable)
        _update_fx_settings(settings=settings,
                            variable=variable,
                            config_user=config_user)
        _update_target_grid(
            variable=variable,
            variables=variables,
            settings=settings,
            config_user=config_user,
        )
        _update_regrid_time(variable, settings)
        ancestors = grouped_ancestors.get(variable['filename'])
        if not ancestors:
            ancestors = _get_ancestors(variable, config_user)
            if config_user.get('skip-nonexistent') and not ancestors:
                logger.info("Skipping: no data found for %s", variable)
                continue
        product = PreprocessorFile(
            attributes=variable,
            settings=settings,
            ancestors=ancestors,
        )
        products.add(product)

    _update_statistic_settings(products, order, config_user['preproc_dir'])

    for product in products:
        product.check()

    return products


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
        config_user=config_user)

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

    logger.info("PreprocessingTask %s created. It will create the files:\n%s",
                task.name, '\n'.join(p.filename for p in task.products))

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
        if not variable.get('force_derivation') and _get_input_files(
                variable, config_user)[0]:
            # No need to derive, just process normally up to derive step
            var = deepcopy(variable)
            append(group_prefix, var)
        else:
            # Process input data needed to derive variable
            required_vars = get_required(variable['short_name'],
                                         variable['project'])
            for var in required_vars:
                _augment(var, variable)
                _add_cmor_info(var, override=True)
                files = _get_input_files(var, config_user)[0]
                if var.get('optional') and not files:
                    logger.info(
                        "Skipping: no data found for %s which is marked as "
                        "'optional'", var)
                else:
                    append(group_prefix, var)

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
        self._cfg = deepcopy(config_user)
        self._cfg['write_ncl_interface'] = self._need_ncl(
            raw_recipe['diagnostics'])
        self._filename = os.path.basename(recipe_file)
        self._preprocessors = raw_recipe.get('preprocessors', {})
        if 'default' not in self._preprocessors:
            self._preprocessors['default'] = {}
        self.diagnostics = self._initialize_diagnostics(
            raw_recipe['diagnostics'], raw_recipe.get('datasets', []))
        self.entity = self._initialize_provenance(
            raw_recipe.get('documentation', {}))
        self.tasks = self.initialize_tasks() if initialize_tasks else None

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
        for key in doc:
            if key in TAGS:
                doc[key] = replace_tags(key, doc[key])

        return get_recipe_provenance(doc, self._filename)

    def _initialize_diagnostics(self, raw_diagnostics, raw_datasets):
        """Define diagnostics in recipe."""
        logger.debug("Retrieving diagnostics from recipe")

        diagnostics = {}

        for name, raw_diagnostic in raw_diagnostics.items():
            diagnostic = {}
            diagnostic['name'] = name
            diagnostic['preprocessor_output'] = \
                self._initialize_preprocessor_output(
                    name,
                    raw_diagnostic.get('variables', {}),
                    raw_datasets +
                    raw_diagnostic.get('additional_datasets', []))
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
    def _expand_ensemble(variables):
        """
        Expand ensemble members to multiple datasets.

        Expansion only supports ensembles defined as strings, not lists.
        """
        expanded = []
        regex = re.compile(r'\(\d+:\d+\)')

        def expand_ensemble(variable):
            ens = variable.get('ensemble', "")
            match = regex.search(ens)
            if match:
                start, end = match.group(0)[1:-1].split(':')
                for i in range(int(start), int(end) + 1):
                    expand = deepcopy(variable)
                    expand['ensemble'] = regex.sub(str(i), ens, 1)
                    expand_ensemble(expand)
            else:
                expanded.append(variable)

        for variable in variables:
            ensemble = variable.get('ensemble', "")
            if isinstance(ensemble, (list, tuple)):
                for elem in ensemble:
                    if regex.search(elem):
                        raise RecipeError(
                            f"In variable {variable}: ensemble expansion "
                            "cannot be combined with ensemble lists")
                expanded.append(variable)
            else:
                expand_ensemble(variable)

        return expanded

    def _initialize_variables(self, raw_variable, raw_datasets):
        """Define variables for all datasets."""
        variables = []

        raw_variable = deepcopy(raw_variable)
        datasets = self._initialize_datasets(
            raw_datasets + raw_variable.pop('additional_datasets', []))
        check.duplicate_datasets(datasets)

        for index, dataset in enumerate(datasets):
            variable = deepcopy(raw_variable)
            variable.update(dataset)

            variable['recipe_dataset_index'] = index
            if 'end_year' in variable and 'max_years' in self._cfg:
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
            required_keys.update({'start_year', 'end_year'})
        for variable in variables:
            if 'institute' not in variable:
                institute = get_institutes(variable)
                if institute:
                    variable['institute'] = institute
            if 'activity' not in variable:
                activity = get_activity(variable)
                if activity:
                    variable['activity'] = activity
            check.variable(variable, required_keys)
        variables = self._expand_ensemble(variables)
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
                    'write_plots',
                    'write_netcdf',
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
                if isinstance(tasks[task_id], DiagnosticTask):
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

    def initialize_tasks(self):
        """Define tasks in recipe."""
        logger.info("Creating tasks from recipe")
        tasks = set()

        priority = 0
        for diagnostic_name, diagnostic in self.diagnostics.items():
            logger.info("Creating tasks for diagnostic %s", diagnostic_name)

            # Create preprocessor tasks
            for variable_group in diagnostic['preprocessor_output']:
                task_name = diagnostic_name + TASKSEP + variable_group
                logger.info("Creating preprocessor task %s", task_name)
                task = _get_preprocessor_task(
                    variables=diagnostic['preprocessor_output']
                    [variable_group],
                    profiles=self._preprocessors,
                    config_user=self._cfg,
                    task_name=task_name,
                )
                for task0 in task.flatten():
                    task0.priority = priority
                tasks.add(task)
                priority += 1

            # Create diagnostic tasks
            for script_name, script_cfg in diagnostic['scripts'].items():
                task_name = diagnostic_name + TASKSEP + script_name
                logger.info("Creating diagnostic task %s", task_name)
                task = DiagnosticTask(
                    script=script_cfg['script'],
                    output_dir=script_cfg['output_dir'],
                    settings=script_cfg['settings'],
                    name=task_name,
                )
                task.priority = priority
                tasks.add(task)
                priority += 1

        check.tasks_valid(tasks)

        # Resolve diagnostic ancestors
        self._resolve_diagnostic_ancestors(tasks)

        # Select only requested tasks
        tasks = get_flattened_tasks(tasks)
        if not self._cfg.get('run_diagnostic'):
            tasks = {t for t in tasks if isinstance(t, PreprocessingTask)}
        if self._cfg.get('diagnostics'):
            names = {t.name for t in tasks}
            selection = set()
            for pattern in self._cfg.get('diagnostics'):
                selection |= set(fnmatch.filter(names, pattern))
            tasks = {t for t in tasks if t.name in selection}

        tasks = get_flattened_tasks(tasks)
        logger.info("These tasks will be executed: %s",
                    ', '.join(t.name for t in tasks))

        # Initialize task provenance
        for task in tasks:
            task.initialize_provenance(self.entity)

        # TODO: check that no loops are created (will throw RecursionError)

        # Return smallest possible set of tasks
        return get_independent_tasks(tasks)

    def __str__(self):
        """Get human readable summary."""
        return '\n\n'.join(str(task) for task in self.tasks)

    def run(self):
        """Run all tasks in the recipe."""
        run_tasks(self.tasks,
                  max_parallel_tasks=self._cfg['max_parallel_tasks'])
