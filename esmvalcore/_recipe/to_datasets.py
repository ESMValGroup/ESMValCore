"""Module that contains functions for reading the `Dataset`s from a recipe."""
from __future__ import annotations

import logging
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Iterator

from esmvalcore.cmor.table import _CMOR_KEYS, _update_cmor_facets
from esmvalcore.config import Session
from esmvalcore.dataset import Dataset, _isglob
from esmvalcore.esgf.facets import FACETS
from esmvalcore.exceptions import RecipeError
from esmvalcore.local import LocalFile, _replace_years_with_timerange
from esmvalcore.preprocessor._derive import get_required
from esmvalcore.preprocessor._io import DATASET_KEYS
from esmvalcore.preprocessor._supplementary_vars import (
    PREPROCESSOR_SUPPLEMENTARIES,
)
from esmvalcore.typing import Facets, FacetValue

from . import check
from ._io import _load_recipe

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


def _check_supplementaries_valid(supplementaries: Iterable[Facets]) -> None:
    """Check that supplementary variables have a short_name."""
    for facets in supplementaries:
        if 'short_name' not in facets:
            raise RecipeError(
                "'short_name' is required for supplementary_variables "
                f"entries, but missing in {facets}")


def _merge_supplementary_dicts(
    var_facets: Iterable[Facets],
    ds_facets: Iterable[Facets],
) -> list[Facets]:
    """Merge the elements of `var_facets` with those in `ds_facets`.

    Both are lists of dicts containing facets
    """
    _check_supplementaries_valid(var_facets)
    _check_supplementaries_valid(ds_facets)
    merged = {}
    for facets in var_facets:
        merged[facets['short_name']] = facets
    for facets in ds_facets:
        short_name = facets['short_name']
        if short_name not in merged:
            merged[short_name] = {}
        merged[short_name].update(facets)
    return list(merged.values())


def _fix_cmip5_fx_ensemble(dataset: Dataset):
    """Automatically correct the wrong ensemble for CMIP5 fx variables."""
    if (dataset.facets.get('project') == 'CMIP5'
            and dataset.facets.get('mip') == 'fx'
            and dataset.facets.get('ensemble') != 'r0i0p0'
            and not dataset.files):
        original_ensemble = dataset['ensemble']
        copy = dataset.copy()
        copy.facets['ensemble'] = 'r0i0p0'
        if copy.files:
            dataset.facets['ensemble'] = 'r0i0p0'
            logger.info("Corrected wrong 'ensemble' from '%s' to '%s' for %s",
                        original_ensemble, dataset['ensemble'],
                        dataset.summary(shorten=True))
            dataset.find_files()


def _get_supplementary_short_names(
    facets: Facets,
    step: str,
) -> list[str]:
    """Get the most applicable supplementary short_names."""
    # Determine if the main variable is an ocean variable.
    var_facets = dict(facets)
    _update_cmor_facets(var_facets)
    realms = var_facets.get('modeling_realm', [])
    if isinstance(realms, (str, Number)):
        realms = [str(realms)]
    ocean_realms = {'ocean', 'seaIce', 'ocnBgchem'}
    is_ocean_variable = any(realm in ocean_realms for realm in realms)

    # Guess the best matching supplementary variable based on the realm.
    short_names = PREPROCESSOR_SUPPLEMENTARIES[step]['variables']
    if set(short_names) == {'areacella', 'areacello'}:
        short_names = ['areacello'] if is_ocean_variable else ['areacella']
    if set(short_names) == {'sftlf', 'sftof'}:
        short_names = ['sftof'] if is_ocean_variable else ['sftlf']

    return short_names


def _append_missing_supplementaries(
    supplementaries: list[Facets],
    facets: Facets,
    settings: dict[str, Any],
) -> None:
    """Append wildcard definitions for missing supplementary variables."""
    steps = [step for step in settings if step in PREPROCESSOR_SUPPLEMENTARIES]

    project: str = facets['project']  # type: ignore
    for step in steps:
        for short_name in _get_supplementary_short_names(facets, step):
            short_names = {f['short_name'] for f in supplementaries}
            if short_name in short_names:
                continue

            supplementary_facets: Facets = {
                facet: '*'
                for facet in FACETS.get(project, ['mip'])
                if facet not in _CMOR_KEYS
            }
            if 'version' in facets:
                supplementary_facets['version'] = '*'
            supplementary_facets['short_name'] = short_name
            supplementaries.append(supplementary_facets)


def _get_dataset_facets_from_recipe(
    variable_group: str,
    recipe_variable: dict[str, Any],
    recipe_dataset: dict[str, Any],
    profiles: dict[str, Any],
    diagnostic_name: str,
    session: Session,
) -> tuple[Facets, list[Facets]]:
    """Read the facets for a single dataset definition from the recipe."""
    facets = deepcopy(recipe_variable)
    facets.pop('additional_datasets', None)
    recipe_dataset = deepcopy(recipe_dataset)

    supplementaries = _merge_supplementary_dicts(
        facets.pop('supplementary_variables', []),
        recipe_dataset.pop('supplementary_variables', []),
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
        diagnostic=diagnostic_name,
        variable_group=variable_group
    )

    preprocessor = facets.get('preprocessor', 'default')
    settings = profiles.get(preprocessor, {})
    _append_missing_supplementaries(supplementaries, facets, settings)
    supplementaries = [
        facets for facets in supplementaries
        if not facets.pop('skip', False)
    ]

    return facets, supplementaries


def _get_facets_from_recipe(
    recipe: dict[str, Any],
    diagnostic_name: str,
    variable_group: str,
    session: Session,
) -> Iterator[tuple[Facets, list[Facets]]]:
    """Read the facets for the detasets of one variable from the recipe."""
    diagnostic = recipe['diagnostics'][diagnostic_name]
    recipe_variable = diagnostic['variables'][variable_group]
    if recipe_variable is None:
        recipe_variable = {}

    recipe_datasets = (recipe.get('datasets', []) +
                       diagnostic.get('additional_datasets', []) +
                       recipe_variable.get('additional_datasets', []))
    check.duplicate_datasets(recipe_datasets, diagnostic_name, variable_group)

    # The NCL interface requires a distinction between variable and
    # dataset keys as defined in the recipe. `DATASET_KEYS` is used to
    # keep track of which keys are part of the dataset.
    DATASET_KEYS.update(key for ds in recipe_datasets for key in ds)

    profiles = recipe.setdefault('preprocessors', {'default': {}})

    for recipe_dataset in recipe_datasets:
        yield _get_dataset_facets_from_recipe(
            variable_group=variable_group,
            recipe_variable=recipe_variable,
            recipe_dataset=recipe_dataset,
            profiles=profiles,
            diagnostic_name=diagnostic_name,
            session=session,
        )


def _get_datasets_for_variable(
    recipe: dict[str, Any],
    diagnostic_name: str,
    variable_group: str,
    session: Session,
) -> list[Dataset]:
    """Read the datasets from a variable definition in the recipe."""
    logger.debug(
        "Populating list of datasets for variable %s in "
        "diagnostic %s", variable_group, diagnostic_name)

    datasets = []
    idx = 0

    for facets, supplementaries in _get_facets_from_recipe(
            recipe,
            diagnostic_name=diagnostic_name,
            variable_group=variable_group,
            session=session,
    ):
        template0 = Dataset(**facets)
        template0.session = session
        for template1 in template0.from_ranges():
            for supplementary_facets in supplementaries:
                template1.add_supplementary(**supplementary_facets)
            for supplementary_ds in template1.supplementaries:
                supplementary_ds.facets.pop('preprocessor', None)
            for dataset in _dataset_from_files(template1):
                dataset['variable_group'] = variable_group
                dataset['diagnostic'] = diagnostic_name
                dataset['recipe_dataset_index'] = idx  # type: ignore
                logger.debug("Found %s", dataset.summary(shorten=True))
                datasets.append(dataset)
                idx += 1

    return datasets


def datasets_from_recipe(
    recipe: Path | str | dict[str, Any],
    session: Session,
) -> list[Dataset]:
    """Read datasets from a recipe."""
    datasets = []

    recipe = _load_recipe(recipe)
    diagnostics = recipe.get('diagnostics') or {}
    for name, diagnostic in diagnostics.items():
        diagnostic_datasets = []
        for variable_group in diagnostic.get('variables', {}):
            variable_datasets = _get_datasets_for_variable(
                recipe,
                diagnostic_name=name,
                variable_group=variable_group,
                session=session,
            )
            diagnostic_datasets.append(variable_datasets)
            datasets.extend(variable_datasets)

        _set_alias(diagnostic_datasets)

    return datasets


def _dataset_from_files(dataset: Dataset) -> list[Dataset]:
    """Replace facet values of '*' based on available files."""
    result: list[Dataset] = []
    errors: list[str] = []

    if any(_isglob(f) for f in dataset.facets.values()):
        logger.debug(
            "Expanding dataset globs for dataset %s, "
            "this may take a while..", dataset.summary(shorten=True))

    representative_datasets = _representative_datasets(dataset)

    # For derived variables, representative_datasets might contain more than
    # one element
    all_datasets: list[list[tuple[dict, Dataset]]] = []
    for representative_dataset in representative_datasets:
        all_datasets.append([])
        for expanded_ds in representative_dataset.from_files():
            updated_facets = {}
            unexpanded_globs = {}
            for key, value in dataset.facets.items():
                if _isglob(value):
                    if (key in expanded_ds.facets and
                            not _isglob(expanded_ds[key])):
                        updated_facets[key] = expanded_ds.facets[key]
                    else:
                        unexpanded_globs[key] = value

            if unexpanded_globs:
                msg = _report_unexpanded_globs(
                    dataset, expanded_ds, unexpanded_globs
                )
                errors.append(msg)
                continue

            new_ds = dataset.copy()
            new_ds.facets.update(updated_facets)
            new_ds.supplementaries = expanded_ds.supplementaries

            all_datasets[-1].append((updated_facets, new_ds))

    # If globs have been expanded, only consider those datasets that contain
    # all necessary input variables if derivation is necessary
    for (updated_facets, new_ds) in all_datasets[0]:
        other_facets = [[d[0] for d in ds] for ds in all_datasets[1:]]
        if all(updated_facets in facets for facets in other_facets):
            result.append(new_ds)
        else:
            logger.debug(
                "Not all necessary input variables to derive '%s' are "
                "available for dataset %s",
                dataset['short_name'],
                updated_facets,
            )

    if errors:
        raise RecipeError("\n".join(errors))

    return result


def _report_unexpanded_globs(
    unexpanded_ds: Dataset,
    expanded_ds: Dataset,
    unexpanded_globs: dict,
) -> str:
    """Get error message for unexpanded globs."""
    msg = (
        "Unable to replace " +
        ", ".join(f"{k}={v}" for k, v in unexpanded_globs.items()) +
        f" by a value for\n{unexpanded_ds}"
    )

    # Set supplementaries to [] to avoid searching for supplementary files
    expanded_ds.supplementaries = []

    if expanded_ds.files:
        if any(isinstance(f, LocalFile) for f in expanded_ds.files):
            paths_msg = "paths to the "
        else:
            paths_msg = ""
        msg = (
            f"{msg}\nDo the {paths_msg}files:\n" +
            "\n".join(
                f"{f} with facets: {f.facets}" for f in expanded_ds.files
            ) +
            "\nprovide the missing facet values?"
        )
    else:
        timerange = expanded_ds.facets.get('timerange')
        patterns = expanded_ds._file_globs
        msg = (
            f"{msg}\nNo files found matching:\n" +
            "\n".join(str(p) for p in patterns) + (  # type:ignore
                f"\nwithin the requested timerange {timerange}."
                if timerange else ""
            )
        )

    return msg


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

    copy = dataset.copy()
    copy.supplementaries = []
    return not copy.files


def _get_input_datasets(dataset: Dataset) -> list[Dataset]:
    """Determine the input datasets needed for deriving `dataset`."""
    facets = dataset.facets
    if not _derive_needed(dataset):
        _fix_cmip5_fx_ensemble(dataset)
        return [dataset]

    # Configure input datasets needed to derive variable
    datasets = []
    required_vars = get_required(facets['short_name'], facets['project'])
    # idea: add option to specify facets in list of dicts that is value of
    # 'derive' in the recipe and use that instead of get_required?
    for input_facets in required_vars:
        input_dataset = dataset.copy()
        keep = {'alias', 'recipe_dataset_index', *dataset.minimal_facets}
        input_dataset.facets = {
            k: v for k, v in input_dataset.facets.items() if k in keep
        }
        input_dataset.facets.update(input_facets)
        input_dataset.augment_facets()
        _fix_cmip5_fx_ensemble(input_dataset)
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


def _representative_datasets(dataset: Dataset) -> list[Dataset]:
    """Find representative datasets for all input variables."""
    copy = dataset.copy()
    copy.supplementaries = []
    representative_datasets = _get_input_datasets(copy)
    for representative_dataset in representative_datasets:
        representative_dataset.supplementaries = dataset.supplementaries
    return representative_datasets
