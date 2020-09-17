"""Module with functions to check a recipe."""
import itertools
import logging
import os
import subprocess
from shutil import which
import re

import yamale

from ._data_finder import get_start_end_year
from ._task import get_flattened_tasks
from .preprocessor import PreprocessingTask, TIME_PREPROCESSORS

logger = logging.getLogger(__name__)


class RecipeError(Exception):
    """Recipe contains an error."""


def ncl_version():
    """Check the NCL version."""
    ncl = which('ncl')
    if not ncl:
        raise RecipeError("Recipe contains NCL scripts, but cannot find "
                          "an NCL installation.")
    try:
        cmd = [ncl, '-V']
        version = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError:
        logger.error("Failed to execute '%s'", ' '.join(' '.join(cmd)))
        raise RecipeError("Recipe contains NCL scripts, but your NCL "
                          "installation appears to be broken.")

    version = version.strip()
    logger.info("Found NCL version %s", version)

    major, minor = (int(i) for i in version.split('.')[:2])
    if major < 6 or (major == 6 and minor < 4):
        raise RecipeError("NCL version 6.4 or higher is required to run "
                          "a recipe containing NCL scripts.")


def recipe_with_schema(filename):
    """Check if the recipe content matches schema."""
    schema_file = os.path.join(os.path.dirname(__file__), 'recipe_schema.yml')
    logger.debug("Checking recipe against schema %s", schema_file)
    recipe = yamale.make_data(filename)
    schema = yamale.make_schema(schema_file)
    yamale.validate(schema, recipe)


def diagnostics(diags):
    """Check diagnostics in recipe."""
    for name, diagnostic in diags.items():
        if 'scripts' not in diagnostic:
            raise RecipeError(
                "Missing scripts section in diagnostic {}".format(name))
        variable_names = tuple(diagnostic.get('variables', {}))
        scripts = diagnostic.get('scripts')
        if scripts is None:
            scripts = {}
        for script_name, script in scripts.items():
            if script_name in variable_names:
                raise RecipeError(
                    "Invalid script name {} encountered in diagnostic {}: "
                    "scripts cannot have the same name as variables.".format(
                        script_name, name))
            if not script.get('script'):
                raise RecipeError(
                    "No script defined for script {} in diagnostic {}".format(
                        script_name, name))


def duplicate_datasets(datasets):
    """Check for duplicate datasets."""
    checked_datasets_ = []
    for dataset in datasets:
        if dataset in checked_datasets_:
            raise RecipeError(
                "Duplicate dataset {} in datasets section".format(dataset))
        checked_datasets_.append(dataset)


def variable(var, required_keys):
    """Check variables as derived from recipe."""
    required = set(required_keys)
    missing = required - set(var)
    if missing:
        raise RecipeError(
            "Missing keys {} from variable {} in diagnostic {}".format(
                missing, var.get('short_name'), var.get('diagnostic')))


def data_availability(input_files, var, dirnames, filenames):
    """Check if the required input data is available."""
    var = dict(var)
    if not input_files:
        var.pop('filename', None)
        logger.error("No input files found for variable %s", var)
        if dirnames and filenames:
            patterns = itertools.product(dirnames, filenames)
            patterns = [os.path.join(d, f) for (d, f) in patterns]
            if len(patterns) == 1:
                msg = f': {patterns[0]}'
            else:
                msg = '\n{}'.format('\n'.join(patterns))
            logger.error("Looked for files matching%s", msg)
        elif dirnames and not filenames:
            logger.error(
                "Looked for files in %s, but did not find any file pattern "
                "to match against", dirnames)
        elif filenames and not dirnames:
            logger.error(
                "Looked for files matching %s, but did not find any existing "
                "input directory", filenames)
        logger.error("Set 'log_level' to 'debug' to get more information")
        raise RecipeError("Missing data")

    # check time avail only for non-fx variables
    if var['frequency'] == 'fx':
        return

    required_years = set(range(var['start_year'], var['end_year'] + 1))
    available_years = set()

    for filename in input_files:
        start, end = get_start_end_year(filename)
        available_years.update(range(start, end + 1))

    missing_years = required_years - available_years
    if missing_years:
        raise RecipeError(
            "No input data available for years {} in files {}".format(
                ", ".join(str(year) for year in missing_years), input_files))


def tasks_valid(tasks):
    """Check that tasks are consistent."""
    filenames = set()
    msg = "Duplicate preprocessor filename {}, please file a bug report."
    for task in get_flattened_tasks(tasks):
        if isinstance(task, PreprocessingTask):
            for product in task.products:
                if product.filename in filenames:
                    raise ValueError(msg.format(product.filename))
                filenames.add(product.filename)


def check_for_temporal_preprocs(profile):
    """Check for temporal operations on fx variables."""
    temp_preprocs = [
        preproc for preproc in profile
        if profile[preproc] and preproc in TIME_PREPROCESSORS
    ]
    if temp_preprocs:
        raise RecipeError(
            "Time coordinate preprocessor step(s) {} not permitted on fx "
            "vars, please remove them from recipe".format(temp_preprocs))


def extract_shape(settings):
    """Check that `extract_shape` arguments are valid."""
    shapefile = settings.get('shapefile', '')
    if not os.path.exists(shapefile):
        raise RecipeError("In preprocessor function `extract_shape`: "
                          f"Unable to find 'shapefile: {shapefile}'")

    valid = {
        'method': {'contains', 'representative'},
        'crop': {True, False},
        'decomposed': {True, False},
    }
    for key in valid:
        value = settings.get(key)
        if not (value is None or value in valid[key]):
            raise RecipeError(
                f"In preprocessor function `extract_shape`: Invalid value "
                f"'{value}' for argument '{key}', choose from "
                "{}".format(', '.join(f"'{k}'".lower() for k in valid[key])))


def valid_multimodel_statistic(statistic):
    """Check that `statistic` is a valid argument for multimodel stats."""
    valid_names = ["mean", "median", "std", "min", "max"]
    valid_patterns = [r"^(p\d{1,2})(\.\d*)?$"]
    if not (statistic in valid_names or
            re.match(r'|'.join(valid_patterns), statistic)):
        raise RecipeError(
            "Invalid value encountered for `statistic` in preprocessor "
            f"`multi_model_statistics`. Valid values are {valid_names} "
            f"or patterns matching {valid_patterns}. Got '{statistic}.'"
        )
