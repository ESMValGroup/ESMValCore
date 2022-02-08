"""Module with functions to check a recipe."""
import itertools
import logging
import os
import re
import subprocess
from pprint import pformat
from shutil import which

import isodate
import yamale

from ._data_finder import get_start_end_year
from .exceptions import InputFilesNotFound, RecipeError
from .preprocessor import TIME_PREPROCESSORS, PreprocessingTask
from .preprocessor._multimodel import STATISTIC_MAPPING

logger = logging.getLogger(__name__)


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
    yamale.validate(schema, recipe, strict=False)


def diagnostics(diags):
    """Check diagnostics in recipe."""
    if diags is None:
        raise RecipeError('The given recipe does not have any diagnostic.')
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


def _log_data_availability_errors(input_files, var, dirnames, filenames):
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


def _group_years(years):
    """Group an iterable of years into easy to read text.

    Example
    -------
    [1990, 1991, 1992, 1993, 2000] -> "1990-1993, 2000"
    """
    years = sorted(years)
    year = years[0]
    previous_year = year
    starts = [year]
    ends = []
    for year in years[1:]:
        if year != previous_year + 1:
            starts.append(year)
            ends.append(previous_year)
        previous_year = year
    ends.append(year)

    ranges = []
    for start, end in zip(starts, ends):
        ranges.append(f"{start}" if start == end else f"{start}-{end}")

    return ", ".join(ranges)


def data_availability(input_files, var, dirnames, filenames, log=True):
    """Check if input_files cover the required years."""
    if log:
        _log_data_availability_errors(input_files, var, dirnames, filenames)

    if not input_files:
        raise InputFilesNotFound(
            f"Missing data for {var['alias']}: {var['short_name']}")

    if var['frequency'] == 'fx':
        # check time availability only for non-fx variables
        return
    start_year = var['start_year']
    end_year = var['end_year']
    required_years = set(range(start_year, end_year + 1, 1))
    available_years = set()

    for filename in input_files:
        start, end = get_start_end_year(filename)
        available_years.update(range(start, end + 1))

    missing_years = required_years - available_years
    if missing_years:
        missing_txt = _group_years(missing_years)

        raise InputFilesNotFound(
            "No input data available for years {} in files:\n{}".format(
                missing_txt, "\n".join(str(f) for f in input_files)))


def tasks_valid(tasks):
    """Check that tasks are consistent."""
    filenames = set()
    msg = "Duplicate preprocessor filename {}, please file a bug report."
    for task in tasks.flatten():
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


def _verify_statistics(statistics, step):
    """Raise error if multi-model statistics cannot be verified."""
    valid_names = ['std'] + list(STATISTIC_MAPPING.keys())
    valid_patterns = [r"^(p\d{1,2})(\.\d*)?$"]

    for statistic in statistics:
        if not (statistic in valid_names
                or re.match(r'|'.join(valid_patterns), statistic)):
            raise RecipeError(
                "Invalid value encountered for `statistic` in preprocessor "
                f"{step}. Valid values are {valid_names} "
                f"or patterns matching {valid_patterns}. Got '{statistic}'.")


def _verify_span_value(span):
    """Raise error if span argument cannot be verified."""
    valid_names = ('overlap', 'full')
    if span not in valid_names:
        raise RecipeError(
            "Invalid value encountered for `span` in preprocessor "
            f"`multi_model_statistics`. Valid values are {valid_names}."
            f"Got {span}.")


def _verify_groupby(groupby):
    """Raise error if groupby arguments cannot be verified."""
    if not isinstance(groupby, list):
        raise RecipeError(
            "Invalid value encountered for `groupby` in preprocessor "
            "`multi_model_statistics`.`groupby` must be defined as a "
            f"list. Got {groupby}.")


def _verify_keep_input_datasets(keep_input_datasets):
    if not isinstance(keep_input_datasets, bool):
        raise RecipeError(
            "Invalid value encountered for `keep_input_datasets`."
            f"Must be defined as a boolean. Got {keep_input_datasets}."
        )


def _verify_arguments(given, expected):
    """Raise error if arguments cannot be verified."""
    for key in given:
        if key not in expected:
            raise RecipeError(
                f"Unexpected keyword argument encountered: {key}. Valid "
                f"keywords are: {expected}.")


def multimodel_statistics_preproc(settings):
    """Check that the multi-model settings are valid."""
    valid_keys = ['span', 'groupby', 'statistics', 'keep_input_datasets']
    _verify_arguments(settings.keys(), valid_keys)

    span = settings.get('span', None)  # optional, default: overlap
    if span:
        _verify_span_value(span)

    groupby = settings.get('groupby', None)  # optional, default: None
    if groupby:
        _verify_groupby(groupby)

    statistics = settings.get('statistics', None)  # required
    if statistics:
        _verify_statistics(statistics, 'multi_model_statistics')

    keep_input_datasets = settings.get('keep_input_datasets', True)
    _verify_keep_input_datasets(keep_input_datasets)


def ensemble_statistics_preproc(settings):
    """Check that the ensemble settings are valid."""
    valid_keys = ['statistics', 'span']
    _verify_arguments(settings.keys(), valid_keys)

    span = settings.get('span', 'overlap')  # optional, default: overlap
    if span:
        _verify_span_value(span)

    statistics = settings.get('statistics', None)
    if statistics:
        _verify_statistics(statistics, 'ensemble_statistics')


def _check_delimiter(timerange):
    if len(timerange) != 2:
        raise RecipeError("Invalid value encountered for `timerange`. "
                          "Valid values must be separated by `/`. "
                          f"Got {timerange} instead.")


def _check_duration_periods(timerange):
    try:
        isodate.parse_duration(timerange[0])
    except ValueError:
        pass
    else:
        try:
            isodate.parse_duration(timerange[1])
        except ValueError:
            pass
        else:
            raise RecipeError("Invalid value encountered for `timerange`. "
                              "Cannot set both the beginning and the end "
                              "as duration periods.")


def _check_format_years(date):
    if date != '*' and not date.startswith('P'):
        if len(date) < 4:
            date = date.zfill(4)
    return date


def _check_timerange_values(date, timerange):
    try:
        isodate.parse_date(date)
    except ValueError:
        try:
            isodate.parse_duration(date)
        except ValueError as exc:
            if date != '*':
                raise RecipeError("Invalid value encountered for `timerange`. "
                                  "Valid value must follow ISO 8601 standard "
                                  "for dates and duration periods, or be "
                                  "set to '*' to load available years. "
                                  f"Got {timerange} instead.") from exc


def valid_time_selection(timerange):
    """Check that `timerange` tag is well defined."""
    if timerange != '*':
        timerange = timerange.split('/')
        _check_delimiter(timerange)
        _check_duration_periods(timerange)
        for date in timerange:
            date = _check_format_years(date)
            _check_timerange_values(date, timerange)


def reference_for_bias_preproc(products):
    """Check that exactly one reference dataset for bias preproc is given."""
    step = 'bias'
    products = {p for p in products if step in p.settings}
    if not products:
        return

    # Check that exactly one dataset contains the facet ``reference_for_bias:
    # true``
    reference_products = []
    for product in products:
        if product.attributes.get('reference_for_bias', False):
            reference_products.append(product)
    if len(reference_products) != 1:
        products_str = [p.filename for p in products]
        if not reference_products:
            ref_products_str = ". "
        else:
            ref_products_str = [p.filename for p in reference_products]
            ref_products_str = f":\n{pformat(ref_products_str)}.\n"
        raise RecipeError(
            f"Expected exactly 1 dataset with 'reference_for_bias: true' in "
            f"products\n{pformat(products_str)},\nfound "
            f"{len(reference_products):d}{ref_products_str}Please also "
            f"ensure that the reference dataset is not excluded with the "
            f"'exclude' option")
