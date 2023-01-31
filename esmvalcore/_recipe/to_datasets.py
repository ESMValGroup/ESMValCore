"""Module that contains functions for reading the `Dataset`s from a recipe."""
from __future__ import annotations

import logging
from copy import deepcopy
from itertools import groupby
from typing import Any, Iterable

from esmvalcore.cmor.table import _update_cmor_facets
from esmvalcore.config import Session
from esmvalcore.dataset import Dataset, _isglob
from esmvalcore.exceptions import RecipeError
from esmvalcore.local import _replace_years_with_timerange
from esmvalcore.preprocessor._derive import get_required
from esmvalcore.preprocessor._io import DATASET_KEYS
from esmvalcore.typing import Facets, FacetValue

from . import check

logger = logging.getLogger(__name__)

_ALIAS_INFO_KEYS = (
    'project',
    'activity',
    'driver',
    'dataset',
    'exp',
    'sub_experiment',
    'ensemble',
    'version',
)
"""List of keys to be used to compose the alias, ordered by priority."""


def _facet_to_str(facet_value: FacetValue) -> str:
    """Get a string representation of a facet value."""
    if isinstance(facet_value, str):
        return facet_value
    if isinstance(facet_value, Iterable):
        return '-'.join(str(v) for v in facet_value)
    return str(facet_value)


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

    for variable in variables:
        for dataset in variable:
            alias = tuple(
                _facet_to_str(dataset.facets.get(key, None))
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


def _check_ancillaries_valid(ancillaries: Iterable[Facets]) -> None:
    """Check that ancillary variables have a short_name."""
    for facets in ancillaries:
        if 'short_name' not in facets:
            raise RecipeError(
                "'short_name' is required for ancillary_variables "
                f"entries, but missing in {facets}")


def _merge_ancillary_dicts(
    var_facets: Iterable[Facets],
    ds_facets: Iterable[Facets],
) -> list[Facets]:
    """Merge the elements of `var_facets` with those in `ds_facets`.

    Both are lists of dicts containing facets
    """
    _check_ancillaries_valid(var_facets)
    _check_ancillaries_valid(ds_facets)
    merged = {}
    for facets in var_facets:
        merged[facets['short_name']] = facets
    for facets in ds_facets:
        short_name = facets['short_name']
        if short_name not in merged:
            merged[short_name] = {}
        merged[short_name].update(facets)
    return list(merged.values())


def _get_facets_from_recipe(
    variable_group: str,
    recipe_variable: dict[str, Any],
    recipe_dataset: dict[str, Any],
    session: Session,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the facets for a single dataset definition from the recipe."""
    facets = deepcopy(recipe_variable)
    recipe_dataset = deepcopy(recipe_dataset)

    ancillaries = _merge_ancillary_dicts(
        facets.pop('ancillary_variables', []),
        recipe_dataset.pop('ancillary_variables', []),
    )

    facets.update(recipe_dataset)

    if 'short_name' not in facets:
        facets['short_name'] = variable_group

    # Flaky support for limiting the number of years in a recipe.
    # If we want this to work, it should actually be done based on `timerange`,
    # after any wildcards have been resolved.
    if 'end_year' in facets and session['max_years']:
        facets['end_year'] = min(
            facets['end_year'],
            facets['start_year'] + session['max_years'] - 1)

    # Legacy: support start_year and end_year instead of timerange
    _replace_years_with_timerange(facets)

    # Legacy: support wrong capitalization of obs4MIPs
    if facets['project'] == 'obs4mips':
        logger.warning("Correcting capitalization, project 'obs4mips' "
                       "should be written as 'obs4MIPs'")
        facets['project'] = 'obs4MIPs'

    check.variable(
        facets,
        required_keys=(
            'short_name',
            'mip',
            'dataset',
            'project',
        ),
    )

    return facets, ancillaries


def _get_all_datasets_for_variable(
    diagnostic: str,
    variable_group: str,
    recipe_variable: dict[str, Any],
    recipe_datasets: Iterable[dict[str, Any]],
    session: Session,
) -> list[Dataset]:
    """Read the datasets from a variable definition in the recipe."""
    datasets = []
    idx = 0

    for recipe_dataset in recipe_datasets:
        facets, ancillaries = _get_facets_from_recipe(
            variable_group=variable_group,
            recipe_variable=recipe_variable,
            recipe_dataset=recipe_dataset,
            session=session,
        )

        dataset = Dataset(**facets)
        dataset.session = session
        dataset['diagnostic'] = diagnostic
        dataset['variable_group'] = variable_group

        for dataset1 in dataset.from_ranges():
            for ancillary_facets in ancillaries:
                dataset1.add_ancillary(**ancillary_facets)
            for ancillary_ds in dataset1.ancillaries:
                ancillary_ds.facets.pop('preprocessor', None)
            for dataset2 in _from_representative_files(dataset1):
                dataset2['recipe_dataset_index'] = idx  # type: ignore
                datasets.append(dataset2)
                idx += 1

    return datasets


def datasets_from_recipe(
    recipe: dict[str, Any],
    session: Session,
) -> list[Dataset]:
    """Read datasets from a recipe."""
    datasets = []

    recipe = deepcopy(recipe)
    diagnostics = recipe.get('diagnostics') or {}
    for name, diagnostic in diagnostics.items():
        diagnostic_datasets = []
        for variable_group in diagnostic.get('variables', {}):

            # Read variable from recipe
            logger.debug(
                "Populating list of datasets for variable %s in "
                "diagnostic %s", variable_group, name)
            recipe_variable = diagnostic['variables'][variable_group]
            if recipe_variable is None:
                recipe_variable = {}

            # Read datasets from recipe
            recipe_datasets = (recipe.get('datasets', []) +
                               diagnostic.get('additional_datasets', []) +
                               recipe_variable.pop('additional_datasets', []))
            # The NCL interface requires a distinction between variable and
            # dataset keys as defined in the recipe. `DATASET_KEYS` is used to
            # keep track of which keys are part of the dataset.
            DATASET_KEYS.update(key for ds in recipe_datasets for key in ds)
            check.duplicate_datasets(recipe_datasets, name, variable_group)

            variable_datasets = _get_all_datasets_for_variable(
                diagnostic=name,
                variable_group=variable_group,
                recipe_variable=recipe_variable,
                recipe_datasets=recipe_datasets,
                session=session,
            )
            diagnostic_datasets.append(variable_datasets)
            datasets.extend(variable_datasets)

        _set_alias(diagnostic_datasets)

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
            logger.info(
                "For dataset %s: only using ancillary dataset %s, "
                "ignoring duplicate ancillary datasets\n%s",
                dataset.summary(shorten=True),
                ancillary_ds.summary(shorten=True),
                "\n".join(ds.summary(shorten=True) for ds in group[1:]),
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


def _get_input_datasets(dataset: Dataset) -> list[Dataset]:
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
        _update_cmor_facets(input_dataset.facets, override=True)
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


def _representative_dataset(dataset: Dataset) -> Dataset:
    """Find a representative dataset that has files available."""
    datasets = _get_input_datasets(dataset)
    representative_dataset = datasets[0]
    return representative_dataset


def _from_representative_files(dataset: Dataset) -> list[Dataset]:
    """Replace facet values of '*' based on available files."""
    result: list[Dataset] = []
    errors = []

    if any(_isglob(f) for f in dataset.facets.values()) or any(
            _isglob(f) for ds in dataset.ancillaries
            for f in ds.facets.values()):
        logger.debug(
            "Expanding dataset globs for dataset %s, "
            "this may take a while..", dataset.summary(shorten=True))

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
            msg = ("Unable to replace " +
                   ", ".join(f"{k}={v}" for k, v in failed.items()) +
                   f" by a value for\n{dataset}")
            if repr_ds.files:
                msg = (f"{msg}\nDo the paths to:\n" +
                       "\n".join(str(f) for f in repr_ds.files) +
                       "\ncontain the missing facet values?")
            else:
                msg = (f"{msg}\nNo files found matching:\n" +
                       "\n".join(str(p)
                                 for p in repr_ds._file_globs))  # type:ignore
            errors.append(msg)

        new_ds = dataset.copy()
        new_ds.facets.update(updated_facets)
        new_ds.ancillaries = [ds.copy() for ds in repr_ds.ancillaries]
        _clean_ancillaries(new_ds)
        logger.debug("Found %s", new_ds.summary(shorten=True))
        result.append(new_ds)

    if errors:
        raise RecipeError("\n".join(errors))

    return result
