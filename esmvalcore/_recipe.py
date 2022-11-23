"""Recipe parser."""
import fnmatch
import logging
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable

import yaml
from nested_lookup import nested_delete

from . import __version__
from . import _recipe_checks as check
from . import esgf
from ._data_finder import (
    _get_timerange_from_years,
    _parse_period,
    _truncate_dates,
    dates_to_timerange,
    get_multiproduct_filename,
    get_output_file,
)
from ._provenance import get_recipe_provenance
from ._task import DiagnosticTask, ResumeTask, TaskSet
from .cmor.table import CMOR_TABLES, _get_facets_from_cmor_table
from .config import Session
from .config._config import TASKSEP, get_project_config
from .config._diagnostics import TAGS
from .dataset import Dataset, _isglob
from .esgf import ESGFFile
from .exceptions import InputFilesNotFound, RecipeError
from .preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
    PreprocessingTask,
    PreprocessorFile,
)
from .preprocessor._ancillary_vars import PREPROCESSOR_ANCILLARIES
from .preprocessor._derive import get_required
from .preprocessor._io import DATASET_KEYS
from .preprocessor._other import _group_products
from .preprocessor._regrid import (
    _spec_to_latlonvals,
    get_cmor_levels,
    get_reference_levels,
    parse_cell_spec,
)
from .types import Facets

logger = logging.getLogger(__name__)

PreprocessorSettings = dict[str, Any]

DOWNLOAD_FILES = set()
"""Use a global variable to keep track of files that need to be downloaded."""

_REQUIRED_KEYS = (
    'short_name',
    'mip',
    'dataset',
    'project',
)

_ALIAS_INFO_KEYS = (
    'project',
    'activity',
    'dataset',
    'exp',
    'ensemble',
    'version',
)
"""List of keys to be used to compose the alias, ordered by priority."""


def read_recipe_file(filename: Path, session):
    """Read a recipe from file."""
    check.recipe_with_schema(filename)
    with open(filename, 'r', encoding='utf-8') as file:
        raw_recipe = yaml.safe_load(file)

    return Recipe(raw_recipe, session, recipe_file=filename)


def _special_name_to_dataset(facets, special_name):
    """Convert special names to dataset names."""
    if special_name in ('reference_dataset', 'alternative_dataset'):
        if special_name not in facets:
            raise RecipeError(
                "Preprocessor {preproc} uses {name}, but {name} is not "
                "defined for variable {short_name} of diagnostic "
                "{diagnostic}".format(
                    preproc=facets['preprocessor'],
                    name=special_name,
                    short_name=facets['short_name'],
                    diagnostic=facets['diagnostic'],
                ))
        special_name = facets[special_name]

    return special_name


def _update_target_levels(dataset, datasets, settings):
    """Replace the target levels dataset name with a filename if needed."""
    levels = settings.get('extract_levels', {}).get('levels')
    if not levels:
        return

    levels = _special_name_to_dataset(dataset.facets, levels)

    # If levels is a dataset name, replace it by a dict with a 'dataset' entry
    if any(levels == d.facets['dataset'] for d in datasets):
        settings['extract_levels']['levels'] = {'dataset': levels}
        levels = settings['extract_levels']['levels']

    if not isinstance(levels, dict):
        return

    if 'cmor_table' in levels and 'coordinate' in levels:
        settings['extract_levels']['levels'] = get_cmor_levels(
            levels['cmor_table'], levels['coordinate'])
    elif 'dataset' in levels:
        dataset_name = levels['dataset']
        if dataset.facets['dataset'] == dataset_name:
            del settings['extract_levels']
        else:
            target_ds = _select_dataset(dataset_name, datasets)
            representative_ds = _representative_dataset(target_ds)
            settings['extract_levels']['levels'] = get_reference_levels(
                representative_ds)


def _update_target_grid(dataset, datasets, settings):
    """Replace the target grid dataset name with a filename if needed."""
    grid = settings.get('regrid', {}).get('target_grid')
    if not grid:
        return

    grid = _special_name_to_dataset(dataset.facets, grid)

    if dataset.facets['dataset'] == grid:
        del settings['regrid']
    elif any(grid == d.facets['dataset'] for d in datasets):
        representative_ds = _representative_dataset(
            _select_dataset(grid, datasets))
        settings['regrid']['target_grid'] = representative_ds
    else:
        # Check that MxN grid spec is correct
        target_grid = settings['regrid']['target_grid']
        if isinstance(target_grid, str):
            parse_cell_spec(target_grid)
        # Check that cdo spec is correct
        elif isinstance(target_grid, dict):
            _spec_to_latlonvals(**target_grid)


def _update_regrid_time(dataset, settings):
    """Input data frequency automatically for regrid_time preprocessor."""
    regrid_time = settings.get('regrid_time')
    if regrid_time is None:
        return
    frequency = settings.get('regrid_time', {}).get('frequency')
    if not frequency:
        settings['regrid_time']['frequency'] = dataset.facets['frequency']


def _select_dataset(dataset_name, datasets):
    for dataset in datasets:
        if dataset.facets['dataset'] == dataset_name:
            return dataset
    raise RecipeError(
        f"Unable to find matching file for dataset {dataset_name}")


def _limit_datasets(datasets, profile):
    """Try to limit the number of datasets to max_datasets."""
    max_datasets = datasets[0].session['max_datasets']
    if not max_datasets:
        return datasets

    logger.info("Limiting the number of datasets to %s", max_datasets)

    required_datasets = [
        (profile.get('extract_levels') or {}).get('levels'),
        (profile.get('regrid') or {}).get('target_grid'),
        datasets[0].facets.get('reference_dataset'),
        datasets[0].facets.get('alternative_dataset'),
    ]

    limited = [d for d in datasets if d.facets['dataset'] in required_datasets]
    for dataset in datasets:
        if len(limited) >= max_datasets:
            break
        if dataset not in limited:
            limited.append(dataset)

    logger.info("Only considering %s",
                ', '.join(d.facets['alias'] for d in limited))

    return limited


def _derive_needed(dataset: Dataset) -> bool:
    """Check if dataset needs to be derived from other datasets."""
    if not dataset.facets.get('derive'):
        return False
    if dataset.facets.get('force_derivation'):
        return True
    if _isglob(dataset.facets.get('timerange', '')):
        # Our file finding routines are not able to handle globs.
        dataset = dataset.copy()
        dataset.facets.pop('timerange')

    return not dataset.files


def _get_default_settings(dataset):
    """Get default preprocessor settings."""
    session = dataset.session
    facets = dataset.facets

    settings = {}

    if _derive_needed(dataset):
        settings['derive'] = {
            'short_name': facets['short_name'],
            'standard_name': facets['standard_name'],
            'long_name': facets['long_name'],
            'units': facets['units'],
        }

    # Clean up fixed files
    if not session['save_intermediary_cubes']:
        output_file = get_output_file(facets, session.preproc_dir)
        fix_dir = f"{output_file.with_suffix('')}_fixed"
        # TODO: check that fixed files are also removed for derived vars
        settings['cleanup'] = {
            'remove': [fix_dir],
        }

    # Configure fx settings
    settings['remove_fx_variables'] = {}

    # Configure saving cubes to file
    settings['save'] = {'compress': session['compress_netcdf']}
    if facets['short_name'] != facets['original_short_name']:
        settings['save']['alias'] = facets['short_name']

    return settings


def _guess_fx_mip(facets: dict, dataset: Dataset):
    """Search mip for fx variable."""
    project = facets.get('project', dataset.facets['project'])
    # check if project in config-developer
    get_project_config(project)

    tables = CMOR_TABLES[project].tables

    # Get all mips that offer that specific fx variable
    mips_with_fx_var = []
    for mip in tables:
        if facets['short_name'] in tables[mip]:
            mips_with_fx_var.append(mip)

    # List is empty -> no table includes the fx variable
    if not mips_with_fx_var:
        raise RecipeError(
            f"Requested fx variable '{facets['short_name']}' not available "
            f"in any CMOR table for '{project}'")

    # Iterate through all possible mips and check if files are available; in
    # case of ambiguity raise an error
    fx_files_for_mips = {}
    for mip in mips_with_fx_var:
        logger.debug("For fx variable '%s', found table '%s'",
                     facets['short_name'], mip)
        fx_dataset = dataset.copy(**facets)
        fx_dataset.ancillaries = []
        fx_dataset.set_facet('mip', mip)
        fx_dataset.facets.pop('timerange', None)
        fx_files = fx_dataset.files
        if fx_files:
            logger.debug("Found fx variables '%s':\n%s", facets['short_name'],
                         pformat(fx_files))
            fx_files_for_mips[mip] = fx_files

    # Dict contains more than one element -> ambiguity
    if len(fx_files_for_mips) > 1:
        raise RecipeError(
            f"Requested fx variable '{facets['short_name']}' for dataset "
            f"'{dataset.facets['dataset']}' of project '{project}' is "
            f"available in more than one CMOR MIP table for "
            f"'{project}': {sorted(fx_files_for_mips)}")

    # Dict is empty -> no files found -> handled at later stage
    if not fx_files_for_mips:
        return mips_with_fx_var[0]

    # Dict contains one element -> ok
    mip = list(fx_files_for_mips)[0]
    return mip


def _get_legacy_ancillary_facets(
    dataset: Dataset,
    settings: PreprocessorSettings,
) -> list[Facets]:
    """Load the ancillary dataset facets from the preprocessor settings."""
    # Update `fx_variables` key in preprocessor settings with defaults
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
    if dataset.facets['project'] != 'obs4MIPs':
        default_fx['area_statistics']['areacello'] = None
        default_fx['mask_landsea']['sftof'] = None
        default_fx['weighting_landsea_fraction']['sftof'] = None

    for step in default_fx:
        if step in settings and 'fx_variables' not in settings[step]:
            settings[step]['fx_variables'] = default_fx[step]

    # Read facets from `fx_variables` key in preprocessor settings
    ancillaries = []
    for step, kwargs in settings.items():
        allowed = PREPROCESSOR_ANCILLARIES.get(step, {}).get('variables', [])
        if 'fx_variables' in kwargs:
            fx_variables = kwargs['fx_variables']

            if fx_variables is None:
                continue

            if isinstance(fx_variables, list):
                result: dict[str, Facets] = {}
                for fx_variable in fx_variables:
                    if isinstance(fx_variable, str):
                        # Legacy legacy method of specifying ancillary variable
                        short_name = fx_variable
                        result[short_name] = {}
                    elif isinstance(fx_variable, dict):
                        short_name = fx_variable['short_name']
                        result[short_name] = fx_variable
                fx_variables = result

            for short_name, facets in fx_variables.items():
                if short_name not in allowed:
                    raise RecipeError(
                        f"Preprocessor function '{step}' does not support "
                        f"ancillary variable '{short_name}'")
                if facets is None:
                    facets = {}
                facets['short_name'] = short_name
                ancillaries.append(facets)

    # Guess the ensemble and mip if they is not specified
    for facets in ancillaries:
        if 'ensemble' not in facets and dataset.facets['project'] == 'CMIP5':
            facets['ensemble'] = 'r0i0p0'
        if 'mip' not in facets:
            facets['mip'] = _guess_fx_mip(facets, dataset)
    return ancillaries


def _add_legacy_ancillary_datasets(dataset: Dataset, settings):
    """Update fx settings depending on the needed method."""
    if dataset.ancillaries:
        # Ancillaries have been defined using the new method.
        return

    logger.info("Using legacy method of adding ancillary variables.")

    legacy_ds = dataset.copy()
    for facets in _get_legacy_ancillary_facets(dataset, settings):
        legacy_ds.add_ancillary(**facets)

    for ancillary_ds in legacy_ds.ancillaries:
        _get_facets_from_cmor_table(ancillary_ds.facets, override=True)
        if ancillary_ds.files:
            dataset.ancillaries.append(ancillary_ds)

    # Remove preprocessor keyword argument `fx_variables`
    for kwargs in settings.values():
        kwargs.pop('fx_variables', None)


def _exclude_dataset(settings, facets, step):
    """Exclude dataset from specific preprocessor step if requested."""
    exclude = {
        _special_name_to_dataset(facets, dataset)
        for dataset in settings[step].pop('exclude', [])
    }
    if facets['dataset'] in exclude:
        settings.pop(step)
        logger.debug("Excluded dataset '%s' from preprocessor step '%s'",
                     facets['dataset'], step)


def _update_weighting_settings(settings, facets):
    """Update settings for the weighting preprocessors."""
    if 'weighting_landsea_fraction' not in settings:
        return
    _exclude_dataset(settings, facets, 'weighting_landsea_fraction')


def _add_to_download_list(dataset):
    """Add the files of `dataset` to `DOWNLOAD_FILES`."""
    for i, file in enumerate(dataset.files):
        if isinstance(file, ESGFFile):
            DOWNLOAD_FILES.add(file)
            dataset.files[i] = file.local_file(dataset.session['download_dir'])


def _schedule_for_download(datasets):
    """Schedule files for download and show the list of files in the log."""
    for dataset in datasets:
        _add_to_download_list(dataset)
        for ancillary_ds in dataset.ancillaries:
            _add_to_download_list(ancillary_ds)

        files = list(dataset.files)
        for ancillary_ds in dataset.ancillaries:
            files.extend(ancillary_ds.files)

        logger.debug(
            "Using input files for variable %s of dataset %s:\n%s",
            dataset.facets['short_name'],
            dataset.facets['alias'].replace('_', ' '),
            '\n'.join(
                f'{f} (will be downloaded)' if not f.exists() else str(f)
                for f in files),
        )


def _check_input_files(input_datasets: list[Dataset],
                       settings: dict[str, Any]):
    """Check that the required input files are available."""
    missing = set()

    for input_dataset in input_datasets:
        try:
            check.data_availability(input_dataset)
        except RecipeError as ex:
            missing.add(ex.message)
        try:
            check.ancillary_availability(
                dataset=input_dataset,
                settings=settings,
            )
        except RecipeError as ex:
            missing.add(ex.message)

    return missing


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


def _update_multi_dataset_settings(facets, settings):
    """Configure multi dataset statistics."""
    for step in MULTI_MODEL_FUNCTIONS:
        if not settings.get(step):
            continue
        # Exclude dataset if requested
        _exclude_dataset(settings, facets, step)


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
            statistic_product = PreprocessorFile(
                filename=filename,
                attributes=statistic_attributes,
                settings=downstream_settings,
            )  # Note that ancestors is set when running the preprocessor func.
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


def _update_extract_shape(settings, session):
    if 'extract_shape' in settings:
        shapefile = settings['extract_shape'].get('shapefile')
        if shapefile:
            if not os.path.exists(shapefile):
                shapefile = os.path.join(
                    session['auxiliary_data_dir'],
                    shapefile,
                )
                settings['extract_shape']['shapefile'] = shapefile
        check.extract_shape(settings['extract_shape'])


def _allow_skipping(dataset: Dataset):
    """Allow skipping of datasets."""
    allow_skipping = all([
        dataset.session['skip_nonexistent'],
        dataset.facets['dataset'] != dataset.facets.get('reference_dataset'),
    ])
    return allow_skipping


def _get_input_datasets(dataset: 'Dataset') -> list['Dataset']:
    """Determine the input datasets needed for deriving `dataset`."""
    facets = dataset.facets
    if not _derive_needed(dataset):
        # No derivation requested or needed
        return [dataset]

    # Configure input datasets needed to derive variable
    datasets = []
    required_vars = get_required(facets['short_name'], facets['project'])
    for input_facets in required_vars:
        input_dataset = dataset.copy(**input_facets)
        # idea: specify facets in list of dicts that is value of 'derive'?
        _get_facets_from_cmor_table(input_dataset.facets, override=True)
        input_dataset.augment_facets()
        if input_facets.get('optional') and not input_dataset.files:
            logger.info(
                "Skipping: no data found for %s which is marked as "
                "'optional'", input_dataset)
        else:
            datasets.append(input_dataset)

    # Check timeranges of available input data.
    timeranges = set()
    for input_dataset in datasets:
        if 'timerange' in input_dataset.facets:
            timeranges.add(input_dataset.facets['timerange'])
    check.differing_timeranges(timeranges, required_vars)

    return datasets


def _representative_dataset(dataset: 'Dataset') -> 'Dataset':
    """Find a representative dataset that has files available."""
    datasets = _get_input_datasets(dataset)
    representative_dataset = datasets[0]
    return representative_dataset


def _set_version(dataset: Dataset, input_datasets: list[Dataset]):
    """Set the 'version' facet based on derivation input datasets."""
    versions = set()
    for in_dataset in input_datasets:
        in_dataset.set_version()
        if version := dataset.facets.get('version'):
            if isinstance(version, list):
                versions.update(version)
            else:
                versions.add(version)
    if versions:
        version = versions.pop() if len(versions) == 1 else sorted(versions)
        dataset.set_facet('version', version)


def _get_preprocessor_products(
    datasets: list[Dataset],
    profile: dict[str, Any],
    order: list[str],
    name: str,
) -> set[PreprocessorFile]:
    """Get preprocessor product definitions for a set of datasets.

    It updates recipe settings as needed by various preprocessors and
    sets the correct ancestry.
    """
    products = set()

    datasets = _limit_datasets(datasets, profile)

    missing_vars: set[str] = set()
    for dataset in datasets:
        dataset.augment_facets()

    for dataset in datasets:
        settings = _get_default_settings(dataset)
        _update_warning_settings(settings, dataset.facets['project'])
        _apply_preprocessor_profile(settings, profile)
        _update_multi_dataset_settings(dataset.facets, settings)
        _update_preproc_functions(settings, dataset, datasets, missing_vars)
        input_datasets = _get_input_datasets(dataset)
        for input_dataset in input_datasets:
            _add_legacy_ancillary_datasets(input_dataset, settings)
        missing = _check_input_files(input_datasets, settings)
        if missing:
            if _allow_skipping(dataset):
                logger.info("Skipping: %s", missing)
            else:
                missing_vars.update(missing)
            continue
        _set_version(dataset, input_datasets)
        _schedule_for_download(input_datasets)
        logger.info("Found input files for %s", dataset.summary(shorten=True))

        filename = get_output_file(dataset.facets, dataset.session.preproc_dir)
        product = PreprocessorFile(
            filename=filename,
            attributes=dataset.facets,
            settings=settings,
            datasets=input_datasets,
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
    preproc_dir = datasets[0].session.preproc_dir
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


def _update_preproc_functions(settings, dataset, datasets, missing_vars):
    session = dataset.session
    _update_extract_shape(settings, session)
    _update_weighting_settings(settings, dataset.facets)
    try:
        _update_target_levels(
            dataset=dataset,
            datasets=datasets,
            settings=settings,
        )
    except RecipeError as ex:
        missing_vars.add(ex.message)
    try:
        _update_target_grid(
            dataset=dataset,
            datasets=datasets,
            settings=settings,
        )
    except RecipeError as ex:
        missing_vars.add(ex.message)
    _update_regrid_time(dataset, settings)
    if dataset.facets.get('frequency') == 'fx':
        check.check_for_temporal_preprocs(settings)


def _extract_preprocessor_order(profile):
    """Extract the order of the preprocessing steps from the profile."""
    custom_order = profile.pop('custom_order', False)
    if not custom_order:
        return DEFAULT_ORDER
    order = tuple(p for p in profile if p not in INITIAL_STEPS + FINAL_STEPS)
    return INITIAL_STEPS + order + FINAL_STEPS


def _get_preprocessor_task(datasets, profiles, task_name):
    """Create preprocessor task(s) for a set of datasets."""
    # First set up the preprocessor profile
    facets = datasets[0].facets
    session = datasets[0].session
    if facets['preprocessor'] not in profiles:
        raise RecipeError(
            f"Unknown preprocessor {facets['preprocessor']} in variable "
            f"{facets['variable_group']} of diagnostic {facets['diagnostic']}")
    logger.info("Creating preprocessor '%s' task for variable '%s'",
                facets['preprocessor'], facets['variable_group'])
    profile = deepcopy(profiles[facets['preprocessor']])
    order = _extract_preprocessor_order(profile)

    # Create preprocessor task
    products = _get_preprocessor_products(
        datasets=datasets,
        profile=profile,
        order=order,
        name=task_name,
    )

    if not products:
        raise RecipeError(
            f"Did not find any input data for task {task_name}")

    task = PreprocessingTask(
        products=products,
        name=task_name,
        order=order,
        debug=session['save_intermediary_cubes'],
        write_ncl_interface=session['write_ncl_interface'],
    )

    logger.info("PreprocessingTask %s created.", task.name)
    logger.debug("PreprocessingTask %s will create the files:\n%s", task.name,
                 '\n'.join(str(p.filename) for p in task.products))

    return task


class Recipe:
    """Recipe object."""

    info_keys = ('project', 'activity', 'dataset', 'exp', 'ensemble',
                 'version')
    """List of keys to be used to compose the alias, ordered by priority."""

    def __init__(self, raw_recipe, session, recipe_file: Path):
        """Parse a recipe file into an object."""
        # Clear the global variable containing the set of files to download
        DOWNLOAD_FILES.clear()
        self._download_files: set[ESGFFile] = set()
        self.session = session
        self.session['write_ncl_interface'] = self._need_ncl(
            raw_recipe['diagnostics'])
        self.datasets = Dataset.from_recipe(recipe_file, session)
        self._raw_recipe = raw_recipe
        self._filename = Path(recipe_file.name)
        self._preprocessors = raw_recipe.get('preprocessors', {})
        if 'default' not in self._preprocessors:
            self._preprocessors['default'] = {}
        self.diagnostics = self._initialize_diagnostics(
            raw_recipe['diagnostics'])
        self.entity = self._initialize_provenance(
            raw_recipe.get('documentation', {}))
        try:
            self.tasks = self.initialize_tasks()
        except RecipeError as exc:
            self._log_recipe_errors(exc)
            raise

    def _log_recipe_errors(self, exc):
        """Log a message with recipe errors."""
        logger.error(exc.message)
        for task in exc.failed_tasks:
            logger.error(task.message)

        if self.session['offline'] and any(
                isinstance(err, InputFilesNotFound)
                for err in exc.failed_tasks):
            logger.error(
                "Not all input files required to run the recipe could be"
                " found.")
            logger.error(
                "If the files are available locally, please check"
                " your `rootpath` and `drs` settings in your user "
                "configuration file %s", self.session['config_file'])
            logger.error(
                "To automatically download the required files to "
                "`download_dir: %s`, set `offline: false` in %s or run the "
                "recipe with the extra command line argument --offline=False",
                self.session['download_dir'],
                self.session['config_file'],
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

    def _initialize_diagnostics(self, raw_diagnostics):
        """Define diagnostics in recipe."""
        logger.debug("Retrieving diagnostics from recipe")
        check.diagnostics(raw_diagnostics)

        diagnostics = {}

        for name, raw_diagnostic in raw_diagnostics.items():
            diagnostic = {}
            diagnostic['name'] = name
            diagnostic['datasets'] = [
                ds for ds in self.datasets if ds.facets['diagnostic'] == name
            ]
            variable_names = tuple(raw_diagnostic.get('variables', {}))
            diagnostic['scripts'] = self._initialize_scripts(
                name, raw_diagnostic.get('scripts'), variable_names)
            for key in ('themes', 'realms'):
                if key in raw_diagnostic:
                    for script in diagnostic['scripts'].values():
                        script['settings'][key] = raw_diagnostic[key]
            diagnostics[name] = diagnostic

        return diagnostics

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
                settings[dir_name] = os.path.join(
                    getattr(self.session, dir_name), diagnostic_name,
                    script_name)
            # Copy other settings
            if self.session['write_ncl_interface']:
                settings['exit_on_ncl_warning'] = self.session[
                    'exit_on_warning']
            for key in (
                    'output_file_type',
                    'log_level',
                    'profile_diagnostic',
                    'auxiliary_data_dir',
            ):
                settings[key] = self.session[key]

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
                                "Could not find any ancestors matching "
                                f"'{id_glob}'.")
                        logger.debug("Pattern %s matches %s", id_glob,
                                     ancestor_ids)
                        ancestors.extend(tasks[a] for a in ancestor_ids)
                    tasks[task_id].ancestors = ancestors

    def _get_tasks_to_run(self):
        """Get tasks filtered and add ancestors if needed."""
        tasknames_to_run = self.session['diagnostics']
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

        if self.session['run_diagnostic']:
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

    def _create_preprocessor_tasks(self, diagnostic_name, diagnostic,
                                   tasknames_to_run, any_diag_script_is_run):
        """Create preprocessor tasks."""
        tasks = []
        failed_tasks = []
        for variable_group, datasets in groupby(
                diagnostic['datasets'],
                key=lambda ds: ds.facets['variable_group']):
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
            for resume_dir in self.session['resume_from']:
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
                        self.session.preproc_dir,
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
                        datasets=list(datasets),
                        profiles=self._preprocessors,
                        task_name=task_name,
                    )
                except RecipeError as ex:
                    failed_tasks.append(ex)
                else:
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
        if self.session['run_diagnostic']:
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
        if not self.tasks:
            raise RecipeError('No tasks to run!')
        self.write_filled_recipe()

        # Download required data
        if not self.session['offline']:
            esgf.download(self._download_files, self.session['download_dir'])

        self.tasks.run(max_parallel_tasks=self.session['max_parallel_tasks'])
        self.write_html_summary()

    def get_output(self) -> dict:
        """Return the paths to the output plots and data.

        Returns
        -------
        product_filenames : dict
            Lists of products/attributes grouped by task.
        """
        output = {}

        output['session'] = self.session
        output['recipe_filename'] = self._filename
        output['recipe_data'] = self._raw_recipe
        output['task_output'] = {}

        for task in self.tasks.flatten():
            if self.session['remove_preproc_dir'] and isinstance(
                    task, PreprocessingTask):
                # Skip preprocessing tasks that are deleted afterwards
                continue
            output['task_output'][task.name] = task.get_product_attributes()

        return output

    def write_filled_recipe(self):
        """Write copy of recipe with filled wildcards."""
        datasets = [ds for ds in self.datasets if ds.files]
        dataset_recipe = datasets_to_recipe(datasets)

        recipe = deepcopy(self._raw_recipe)
        # Remove dataset sections from recipe
        recipe.pop('datasets', None)
        nested_delete(recipe, 'additional_datasets', in_place=True)

        # Format description nicer
        doc = recipe['documentation']
        if 'description' in doc:
            doc['description'] = doc['description'].strip()

        # Update datasets section
        if 'datasets' in dataset_recipe:
            recipe['datasets'] = dataset_recipe['datasets']

        for diag, dataset_diagnostic in dataset_recipe['diagnostics'].items():
            diagnostic = recipe['diagnostics'][diag]
            # Update diagnostic level datasets
            if 'additional_datasets' in dataset_diagnostic:
                additional_datasets = dataset_diagnostic['additional_datasets']
                diagnostic['additional_datasets'] = additional_datasets
            # Update variable level datasets
            if 'variables' in dataset_diagnostic:
                diagnostic['variables'] = dataset_diagnostic['variables']

        filename = self.session.run_dir / f"{self._filename.stem}_filled.yml"
        with filename.open('w', encoding='utf-8') as file:
            yaml.safe_dump(recipe, file, sort_keys=False)

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


def _set_aliases(datasets: list['Dataset']):
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
    variables : list
        for each recipe variable, a list of datasets
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
                for key in _ALIAS_INFO_KEYS)
            datasets_info.add(alias)
            if 'alias' not in dataset.facets:
                dataset.facets['alias'] = alias

    alias = {}
    for info in datasets_info:
        alias[info] = []

    datasets_info = list(datasets_info)
    _get_next_alias(alias, datasets_info, 0)

    for info in datasets_info:
        alias[info] = '_'.join(
            [str(value) for value in alias[info] if value is not None])
        if not alias[info]:
            alias[info] = info[_ALIAS_INFO_KEYS.index('dataset')]

    for variable in variables:
        for dataset in variable:
            dataset.facets['alias'] = alias.get(dataset.facets['alias'],
                                                dataset.facets['alias'])


def _get_next_alias(alias, datasets_info, i):
    if i >= len(_ALIAS_INFO_KEYS):
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
        merged[facets['short_name']] = facets
    for facets in ds_facets:
        if 'short_name' not in facets:
            raise RecipeError(f"{msg} {facets}")
        short_name = facets['short_name']
        if short_name not in merged:
            merged[short_name] = {}
        merged[short_name].update(facets)

    return list(merged.values())


def datasets_from_recipe(recipe: Path, session: Session) -> list[Dataset]:
    datasets = []

    loaded_recipe = yaml.safe_load(recipe.read_text(encoding='utf-8'))
    diagnostics = loaded_recipe.get('diagnostics') or {}
    for name, diagnostic in diagnostics.items():
        for variable_group in diagnostic.get('variables', {}):
            logger.debug(
                "Populating list of datasets for variable %s in "
                "diagnostic %s", variable_group, name)
            # Read variable from recipe
            recipe_variable = diagnostic['variables'][variable_group]
            if recipe_variable is None:
                recipe_variable = {}
            # Read datasets from recipe
            recipe_datasets = (
                loaded_recipe.get('datasets', []) +
                diagnostic.get('additional_datasets', []) +
                recipe_variable.get('additional_datasets', []))
            check.datasets(recipe_datasets, name, variable_group)

            idx = 0
            for recipe_dataset in recipe_datasets:
                DATASET_KEYS.union(recipe_dataset)
                recipe_dataset = deepcopy(recipe_dataset)
                facets = deepcopy(recipe_variable)
                facets.pop('additional_datasets', None)
                for key, value in recipe_dataset.items():
                    if key == 'ancillary_variables' and key in facets:
                        _merge_ancillary_dicts(facets[key], value)
                    else:
                        facets[key] = value
                # Legacy: support start_year and end_year instead of timerange
                if 'end_year' in facets and session['max_years']:
                    facets['end_year'] = min(
                        facets['end_year'],
                        facets['start_year'] + session['max_years'] - 1)
                _get_timerange_from_years(facets)
                # Legacy: support wrong capitalization of obs4MIPs
                if facets['project'] == 'obs4mips':
                    logger.warning(
                        "Correcting capitalization, project 'obs4mips' "
                        "should be written as 'obs4MIPs'")
                    facets['project'] = 'obs4MIPs'

                persist = set(facets)
                facets['diagnostic'] = name
                facets['variable_group'] = variable_group
                if 'short_name' not in facets:
                    facets['short_name'] = variable_group
                    persist.add('short_name')
                check.variable(facets, required_keys=_REQUIRED_KEYS)
                preprocessor = str(facets.pop('preprocessor', 'default'))
                ancillaries = facets.pop('ancillary_variables', [])
                dataset = Dataset()
                dataset.session = session
                for key, value in facets.items():
                    dataset.set_facet(key, value, key in persist)
                dataset.set_facet('preprocessor', preprocessor,
                                  preprocessor != 'default')
                for dataset1 in dataset.from_ranges():
                    for ancillary_facets in ancillaries:
                        dataset1.add_ancillary(**ancillary_facets)
                    for ancillary_ds in dataset1.ancillaries:
                        ancillary_ds.facets.pop('preprocessor')
                    for dataset2 in _from_representative_files(dataset1):
                        dataset2.facets[
                            'recipe_dataset_index'] = idx  # type: ignore
                        datasets.append(dataset2)
                        idx += 1

    _set_aliases(datasets)

    return datasets


def _clean_ancillaries(dataset: Dataset) -> None:
    """Ignore duplicate and not expanded ancillary variables."""

    def match(ancillary_ds: Dataset) -> int:
        """Compute match of ancillary dataset with main dataset."""
        score = 0
        for key, value in dataset.facets.items():
            if key in ancillary_ds.facets:
                if ancillary_ds.facets[key] == value:
                    score += 1
        return score

    ancillaries = []
    for _, duplicates in groupby(dataset.ancillaries,
                                 key=lambda ds: ds['short_name']):
        group = sorted(duplicates, key=match, reverse=True)
        ancillary_ds = group[0]
        if len(group) > 1:
            logger.warning(
                "For dataset %s: only using ancillary dataset %s, "
                "ignoring duplicate ancillary datasets\n%s",
                dataset.summary(shorten=True),
                ancillary_ds.summary(shorten=True),
                "\n".join(
                    ds.summary(shorten=True) for ds in group[1:]),
            )
        if any(_isglob(v) for v in ancillary_ds.facets.values()):
            logger.warning(
                "For dataset %s: ignoring ancillary dataset %s, "
                "unable to expand wildcards.",
                dataset.summary(shorten=True),
                ancillary_ds.summary(shorten=True),
            )
        else:
            ancillaries.append(ancillary_ds)
    dataset.ancillaries = ancillaries


def _from_representative_files(dataset: Dataset) -> list[Dataset]:
    """Replace facet values of '*' based on available files."""
    logger.info("Expanding dataset globs in recipe, this may take a while..")
    result: list[Dataset] = []
    errors = []
    repr_dataset = _representative_dataset(dataset)

    for repr_ds in repr_dataset.from_files():
        updated_facets = {}
        failed = {}
        for key, value in dataset.facets.items():
            if _isglob(value):
                if key in repr_ds.facets and not _isglob(repr_ds[key]):
                    updated_facets[key] = repr_ds.facets[key]
                else:
                    failed[key] = value

        if failed:
            errors.append("Unable to replace " +
                          ", ".join(f"{k}={v}" for k, v in failed.items()) +
                          f" by a value for {dataset}. Do the paths to:\n" +
                          "\n".join(str(f) for f in repr_ds.files) +
                          "\ncontain the facet values?")

        new_ds = dataset.copy()
        new_ds.facets.update(updated_facets)
        new_ds.ancillaries = [ds.copy() for ds in repr_ds.ancillaries]
        _clean_ancillaries(new_ds)
        logger.debug("Using ancillary dataset %s", new_ds)
        result.append(new_ds)

    if errors:
        raise RecipeError("\n".join(errors))

    return result


def datasets_to_recipe(datasets: Iterable[Dataset]) -> dict[str, Any]:

    diagnostics: dict[str, dict[str, Any]] = {}

    for dataset in datasets:
        if 'diagnostic' not in dataset.facets:
            raise RecipeError(
                f"'diagnostic' facet missing from dataset {dataset},"
                "unable to convert to recipe.")
        diagnostic_name: str = dataset.facets['diagnostic']  # type: ignore
        if diagnostic_name not in diagnostics:
            diagnostics[diagnostic_name] = {'variables': {}}
        variables = diagnostics[diagnostic_name]['variables']
        if 'variable_group' in dataset.facets:
            variable_group = dataset.facets['variable_group']
        else:
            variable_group = dataset.facets['short_name']
        if variable_group not in variables:
            variables[variable_group] = {'additional_datasets': []}
        facets: dict[str, Any] = dataset.minimal_facets
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

    # Move identical facets from dataset to variable
    for diagnostic in diagnostics.values():
        diagnostic['variables'] = {
            variable_group: group_identical_facets(variable)
            for variable_group, variable in diagnostic['variables'].items()
        }

    # TODO: make recipe look nice
    # - deduplicate by moving datasets up from variable to diagnostic to recipe
    # - remove variable_group if the same as short_name
    # - remove automatically added facets

    recipe = {'diagnostics': diagnostics}
    return recipe


def group_identical_facets(variable: dict[str, Any]) -> dict[str, Any]:
    """Move identical facets from datasets to variable."""
    result = dict(variable)
    dataset_facets = result.pop('additional_datasets')
    variable_keys = [
        k for k, v in dataset_facets[0].items()
        if k != 'dataset'  # keep at least one key in every dataset
        and all((k, v) in d.items() for d in dataset_facets[1:])
    ]
    result.update(
        (k, v)
        for k, v in dataset_facets[0].items() if k in variable_keys)
    result['additional_datasets'] = [{
        k: v
        for k, v in d.items() if k not in variable_keys
    } for d in dataset_facets]
    return result
