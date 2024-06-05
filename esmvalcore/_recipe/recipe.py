"""Recipe parser."""
from __future__ import annotations

import fnmatch
import logging
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml

from esmvalcore import __version__, esgf
from esmvalcore._provenance import get_recipe_provenance
from esmvalcore._task import DiagnosticTask, ResumeTask, TaskSet
from esmvalcore.config._config import TASKSEP
from esmvalcore.config._diagnostics import TAGS
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import InputFilesNotFound, RecipeError
from esmvalcore.local import (
    _dates_to_timerange,
    _get_multiproduct_filename,
    _get_output_file,
    _parse_period,
    _truncate_dates,
)
from esmvalcore.preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
    PreprocessingTask,
    PreprocessorFile,
)
from esmvalcore.preprocessor._area import _update_shapefile_path
from esmvalcore.preprocessor._multimodel import _get_stat_identifier
from esmvalcore.preprocessor._regrid import (
    _spec_to_latlonvals,
    get_cmor_levels,
    get_reference_levels,
    parse_cell_spec,
)
from esmvalcore.preprocessor._shared import _group_products

from . import check
from .from_datasets import datasets_to_recipe
from .to_datasets import (
    _derive_needed,
    _get_input_datasets,
    _representative_datasets,
)

logger = logging.getLogger(__name__)

PreprocessorSettings = Dict[str, Any]

DOWNLOAD_FILES = set()
"""Use a global variable to keep track of files that need to be downloaded."""

USED_DATASETS = []
"""Use a global variable to keep track of datasets that are actually used."""


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
                "Preprocessor '{preproc}' uses '{name}', but '{name}' is not "
                "defined for variable '{variable_group}' of diagnostic "
                "'{diagnostic}'.".format(
                    preproc=facets['preprocessor'],
                    name=special_name,
                    variable_group=facets['variable_group'],
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
            representative_ds = _representative_datasets(target_ds)[0]
            check.data_availability(representative_ds)
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
        representative_ds = _representative_datasets(
            _select_dataset(grid, datasets))[0]
        check.data_availability(representative_ds)
        settings['regrid']['target_grid'] = representative_ds
    else:
        # Check that MxN grid spec is correct
        target_grid = settings['regrid']['target_grid']
        if isinstance(target_grid, str):
            parse_cell_spec(target_grid)
        # Check that cdo spec is correct
        elif isinstance(target_grid, dict):
            _spec_to_latlonvals(**target_grid)


def _update_regrid_time(dataset: Dataset, settings: dict) -> None:
    """Input data frequency automatically for regrid_time preprocessor."""
    if 'regrid_time' not in settings:
        return
    if 'frequency' not in settings['regrid_time']:
        settings['regrid_time']['frequency'] = dataset.facets['frequency']


def _select_dataset(dataset_name, datasets):
    for dataset in datasets:
        if dataset.facets['dataset'] == dataset_name:
            return dataset
    diagnostic = datasets[0].facets['diagnostic']
    variable_group = datasets[0].facets['variable_group']
    raise RecipeError(
        f"Unable to find dataset '{dataset_name}' in the list of datasets"
        f"for variable '{variable_group}' of diagnostic '{diagnostic}'.")


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

    # Strip supplementary variables before saving
    settings['remove_supplementary_variables'] = {}

    # Configure saving cubes to file
    settings['save'] = {'compress': session['compress_netcdf']}
    if facets['short_name'] != facets['original_short_name']:
        settings['save']['alias'] = facets['short_name']

    return settings


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
        if isinstance(file, esgf.ESGFFile):
            DOWNLOAD_FILES.add(file)
            dataset.files[i] = file.local_file(dataset.session['download_dir'])


def _schedule_for_download(datasets):
    """Schedule files for download."""
    for dataset in datasets:
        _add_to_download_list(dataset)
        for supplementary_ds in dataset.supplementaries:
            _add_to_download_list(supplementary_ds)


def _log_input_files(datasets: Iterable[Dataset]) -> None:
    """Show list of files in log (including supplementaries)."""
    for dataset in datasets:
        # Only log supplementary variables if present
        supplementary_files_str = ""
        if dataset.supplementaries:
            for sup_ds in dataset.supplementaries:
                supplementary_files_str += (
                    f"\nwith files for supplementary variable "
                    f"{sup_ds['short_name']}:\n{_get_files_str(sup_ds)}"
                )

        logger.debug(
            "Using input files for variable %s of dataset %s:\n%s%s",
            dataset.facets['short_name'],
            dataset.facets['alias'].replace('_', ' '),  # type: ignore
            _get_files_str(dataset),
            supplementary_files_str
        )


def _get_files_str(dataset: Dataset) -> str:
    """Get nice string representation of all files of a dataset."""
    return '\n'.join(
        f'  {f}' if f.exists()  # type: ignore
        else f'  {f} (will be downloaded)' for f in dataset.files
    )


def _check_input_files(input_datasets: Iterable[Dataset]) -> set[str]:
    """Check that the required input files are available."""
    missing = set()

    for input_dataset in input_datasets:
        for dataset in [input_dataset] + input_dataset.supplementaries:
            try:
                check.data_availability(dataset)
            except RecipeError as exc:
                missing.add(exc.message)

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
            attributes['timerange'] = _dates_to_timerange(start, end)
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

            attributes['timerange'] = _dates_to_timerange(start_date, end_date)

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

        statistics = settings.get('statistics', [])
        for statistic in statistics:
            statistic_attributes = dict(common_attributes)
            stat_id = _get_stat_identifier(statistic)
            statistic_attributes[step] = _get_tag(step, identifier, stat_id)
            statistic_attributes.setdefault('alias',
                                            statistic_attributes[step])
            statistic_attributes.setdefault('dataset',
                                            statistic_attributes[step])
            filename = _get_multiproduct_filename(statistic_attributes,
                                                  preproc_dir)
            statistic_product = PreprocessorFile(
                filename=filename,
                attributes=statistic_attributes,
                settings=downstream_settings,
            )  # Note that ancestors is set when running the preprocessor func.
            output_products.add(statistic_product)
            relevant_settings['output_products'][identifier][
                stat_id] = statistic_product

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
            shapefile = _update_shapefile_path(shapefile, session=session)
            settings['extract_shape']['shapefile'] = shapefile
        check.extract_shape(settings['extract_shape'])


def _allow_skipping(dataset: Dataset):
    """Allow skipping of datasets."""
    allow_skipping = all([
        dataset.session['skip_nonexistent'],
        dataset.facets['dataset'] != dataset.facets.get('reference_dataset'),
    ])
    return allow_skipping


def _set_version(dataset: Dataset, input_datasets: list[Dataset]):
    """Set the 'version' facet based on derivation input datasets."""
    versions = set()
    for in_dataset in input_datasets:
        in_dataset.set_version()
        if version := in_dataset.facets.get('version'):
            if isinstance(version, list):
                versions.update(version)
            else:
                versions.add(version)
    if versions:
        version = versions.pop() if len(versions) == 1 else sorted(versions)
        dataset.set_facet('version', version)
    for supplementary_ds in dataset.supplementaries:
        supplementary_ds.set_version()


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
        _apply_preprocessor_profile(settings, profile)
        _update_multi_dataset_settings(dataset.facets, settings)
        _update_preproc_functions(settings, dataset, datasets, missing_vars)
        check.preprocessor_supplementaries(dataset, settings)
        input_datasets = _get_input_datasets(dataset)
        missing = _check_input_files(input_datasets)
        if missing:
            if _allow_skipping(dataset):
                logger.info("Skipping: %s", missing)
            else:
                missing_vars.update(missing)
            continue
        _set_version(dataset, input_datasets)
        USED_DATASETS.append(dataset)
        _schedule_for_download(input_datasets)
        _log_input_files(input_datasets)
        logger.info("Found input files for %s", dataset.summary(shorten=True))

        filename = _get_output_file(
            dataset.facets,
            dataset.session.preproc_dir,
        )
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
    check.reference_for_distance_metric_preproc(products)

    _configure_multi_product_preprocessor(
        products=products,
        preproc_dir=datasets[0].session.preproc_dir,
        profile=profile,
        order=order,
    )

    for product in products:
        _set_start_end_year(product)
        product.check()

    return products


def _configure_multi_product_preprocessor(
    products: Iterable[PreprocessorFile],
    preproc_dir: Path,
    profile: PreprocessorSettings,
    order: Sequence[str],
):
    """Configure preprocessing of ensemble and multimodel statistics."""
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

    for product in multimodel_products | ensemble_products:
        product.check()
        _set_start_end_year(product)


def _set_start_end_year(product: PreprocessorFile) -> None:
    """Set the attributes `start_year` and `end_year`.

    These attributes are used by many diagnostic scripts in ESMValTool.
    """
    if 'timerange' in product.attributes:
        start_year, end_year = _parse_period(product.attributes['timerange'])
        product.attributes['start_year'] = int(str(start_year[0:4]))
        product.attributes['end_year'] = int(str(end_year[0:4]))


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
    except RecipeError as exc:
        missing_vars.add(exc.message)
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
    check.statistics_preprocessors(settings)
    check.regridding_schemes(settings)
    check.bias_type(settings)
    check.metric_type(settings)


def _get_preprocessor_task(datasets, profiles, task_name):
    """Create preprocessor task(s) for a set of datasets."""
    # First set up the preprocessor profile
    facets = datasets[0].facets
    session = datasets[0].session
    preprocessor = facets.get('preprocessor', 'default')
    if preprocessor not in profiles:
        raise RecipeError(
            f"Unknown preprocessor '{preprocessor}' in variable "
            f"{facets['variable_group']} of diagnostic {facets['diagnostic']}")
    logger.info("Creating preprocessor '%s' task for variable '%s'",
                preprocessor, facets['variable_group'])
    profile = deepcopy(profiles[preprocessor])
    order = _extract_preprocessor_order(profile)

    # Create preprocessor task
    products = _get_preprocessor_products(
        datasets=datasets,
        profile=profile,
        order=order,
        name=task_name,
    )

    if not products:
        raise RecipeError(f"Did not find any input data for task {task_name}")

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


def _extract_preprocessor_order(profile):
    """Extract the order of the preprocessing steps from the profile."""
    custom_order = profile.pop('custom_order', False)
    if not custom_order:
        return DEFAULT_ORDER
    if 'derive' not in profile:
        initial_steps = INITIAL_STEPS + ('derive', )
    else:
        initial_steps = INITIAL_STEPS
    order = tuple(p for p in profile if p not in initial_steps + FINAL_STEPS)
    return initial_steps + order + FINAL_STEPS


class Recipe:
    """Recipe object."""

    def __init__(self, raw_recipe, session, recipe_file: Path):
        """Parse a recipe file into an object."""
        # Clear the global variable containing the set of files to download
        DOWNLOAD_FILES.clear()
        USED_DATASETS.clear()
        self._download_files: set[esgf.ESGFFile] = set()
        self.session = session
        self.session['write_ncl_interface'] = self._need_ncl(
            raw_recipe['diagnostics'])
        self._raw_recipe = raw_recipe
        self._filename = Path(recipe_file.name)
        self._preprocessors = raw_recipe.get('preprocessors', {})
        if 'default' not in self._preprocessors:
            self._preprocessors['default'] = {}
        self.datasets = Dataset.from_recipe(recipe_file, session)
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

        if self.session['search_esgf'] == 'never' and any(
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
                "`download_dir: %s`, set `search_esgf: when_missing` or "
                "`search_esgf: always` in %s, or run the recipe with the "
                "extra command line argument --search_esgf=when_missing or "
                "--search_esgf=always",
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
                except RecipeError as exc:
                    failed_tasks.append(exc)
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
        filled_recipe = self.write_filled_recipe()

        # Download required data
        if self.session['search_esgf'] != 'never':
            esgf.download(self._download_files, self.session['download_dir'])

        self.tasks.run(max_parallel_tasks=self.session['max_parallel_tasks'])
        logger.info(
            "Wrote recipe with version numbers and wildcards "
            "to:\nfile://%s", filled_recipe)
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

        for task in sorted(self.tasks.flatten(), key=lambda t: t.priority):
            if self.session['remove_preproc_dir'] and isinstance(
                    task, PreprocessingTask):
                # Skip preprocessing tasks that are deleted afterwards
                continue
            output['task_output'][task.name] = task.get_product_attributes()

        return output

    def write_filled_recipe(self):
        """Write copy of recipe with filled wildcards."""
        recipe = datasets_to_recipe(USED_DATASETS, self._raw_recipe)
        filename = self.session.run_dir / f"{self._filename.stem}_filled.yml"
        with filename.open('w', encoding='utf-8') as file:
            yaml.safe_dump(recipe, file, sort_keys=False)
        logger.info(
            "Wrote recipe with version numbers and wildcards "
            "to:\nfile://%s", filename)
        return filename

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
