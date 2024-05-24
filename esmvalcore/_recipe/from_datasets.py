"""Functions for creating/updating a recipe with `Dataset`s."""
from __future__ import annotations

import itertools
import logging
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence

from nested_lookup import nested_delete

from esmvalcore.exceptions import RecipeError

from ._io import _load_recipe

if TYPE_CHECKING:
    from esmvalcore.dataset import Dataset

logger = logging.getLogger(__name__)

Recipe = Dict[str, Any]
Facets = Dict[str, Any]


def _datasets_to_raw_recipe(datasets: Iterable[Dataset]) -> Recipe:
    """Convert datasets to a recipe dict."""
    diagnostics: dict[str, dict[str, Any]] = {}

    for dataset in datasets:
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
        facets.pop('diagnostic', None)
        if facets['short_name'] == variable_group:
            facets.pop('short_name')
        if dataset.supplementaries:
            facets['supplementary_variables'] = []
        for supplementary in dataset.supplementaries:
            anc_facets = {}
            for key, value in supplementary.minimal_facets.items():
                if facets.get(key) != value:
                    anc_facets[key] = value
            facets['supplementary_variables'].append(anc_facets)
        variables[variable_group]['additional_datasets'].append(facets)

    recipe = {'diagnostics': diagnostics}
    return recipe


def _datasets_to_recipe(datasets: Iterable[Dataset]) -> Recipe:
    """Convert datasets to a condensed recipe dict."""
    for dataset in datasets:
        if 'diagnostic' not in dataset.facets:
            raise RecipeError(f"'diagnostic' facet missing from {dataset},"
                              "unable to convert to recipe.")

    recipe = _datasets_to_raw_recipe(datasets)
    diagnostics = recipe['diagnostics'].values()

    # Group ensemble members
    for diagnostic in diagnostics:
        for variable in diagnostic['variables'].values():
            variable['additional_datasets'] = _group_ensemble_members(
                variable['additional_datasets'])

    # Move identical facets from dataset to variable
    for diagnostic in diagnostics:
        diagnostic['variables'] = {
            variable_group: _group_identical_facets(variable)
            for variable_group, variable in diagnostic['variables'].items()
        }

    # Deduplicate by moving datasets up from variable to diagnostic to recipe
    recipe = _move_datasets_up(recipe)

    return recipe


def _move_datasets_up(recipe: Recipe) -> Recipe:
    """Move datasets from variable to diagnostic to recipe."""
    # Move `additional_datasets` from variable to diagnostic level
    for diagnostic in recipe['diagnostics'].values():
        _move_one_level_up(diagnostic, 'variables', 'additional_datasets')

    # Move `additional_datasets` from diagnostic to `datasets` at recipe level
    _move_one_level_up(recipe, 'diagnostics', 'datasets')

    return recipe


def _to_frozen(item):
    """Return a frozen and sorted copy of nested dicts and lists."""
    if isinstance(item, list):
        return tuple(sorted(_to_frozen(elem) for elem in item))
    if isinstance(item, dict):
        return tuple(sorted((k, _to_frozen(v)) for k, v in item.items()))
    return item


def _move_one_level_up(base: dict, level: str, target: str):
    """Move datasets one level up in the recipe."""
    groups = base[level]
    if not groups:
        return

    # Create a mapping from objects that can be hashed to the dicts
    # describing the datasets.
    dataset_mapping = {}
    for name, group in groups.items():
        dataset_mapping[name] = {
            _to_frozen(ds): ds
            for ds in group['additional_datasets']
        }

    # Set datasets that are common to all groups
    first_datasets = next(iter(dataset_mapping.values()))
    common_datasets = set(first_datasets)
    for datasets in dataset_mapping.values():
        common_datasets &= set(datasets)
    base[target] = [
        v for k, v in first_datasets.items() if k in common_datasets
    ]

    # Remove common datasets from groups
    for name, datasets in dataset_mapping.items():
        group = groups[name]
        var_datasets = set(datasets) - common_datasets
        if var_datasets:
            group['additional_datasets'] = [
                v for k, v in datasets.items() if k in var_datasets
            ]
        else:
            group.pop('additional_datasets')


def _group_identical_facets(variable: Mapping[str, Any]) -> Recipe:
    """Move identical facets from datasets to variable."""
    result = dict(variable)
    dataset_facets = result.pop('additional_datasets')
    variable_keys = [
        k for k, v in dataset_facets[0].items()
        if k != 'dataset'  # keep at least one key in every dataset
        and all((k, v) in d.items() for d in dataset_facets[1:])
    ]
    result.update(
        (k, v) for k, v in dataset_facets[0].items() if k in variable_keys)
    result['additional_datasets'] = [{
        k: v
        for k, v in d.items() if k not in variable_keys
    } for d in dataset_facets]
    return result


def _group_ensemble_members(dataset_facets: Iterable[Facets]) -> list[Facets]:
    """Group ensemble members.

    This is the inverse operation of `Dataset.from_ranges` for
    ensembles.
    """

    def grouper(facets):
        return sorted(
            (f, str(v)) for f, v in facets.items() if f != 'ensemble')

    result = []
    dataset_facets = sorted(dataset_facets, key=grouper)
    for _, group_iter in itertools.groupby(dataset_facets, key=grouper):
        group = list(group_iter)
        ensembles = [f['ensemble'] for f in group if 'ensemble' in f]
        group_facets = group[0]
        if not ensembles:
            result.append(dict(group_facets))
        else:
            for ensemble in _group_ensemble_names(ensembles):
                facets = dict(group_facets)
                facets['ensemble'] = ensemble
                result.append(facets)
    return result


def _group_ensemble_names(ensemble_names: Iterable[str]) -> list[str]:
    """Group ensemble names.

    Examples
    --------
    ensemble_names=[
        'r1i1p1',
        'r2i1p1',
        'r3i1p1',
        'r1i1p2',
    ]
    will return [
        'r(1:3)i1p1',
        'r1i1p2',
    ].
    """
    ensemble_tuples = [
        tuple(int(i) for i in re.findall(r'\d+', ens))
        for ens in ensemble_names
    ]

    ensemble_ranges = _create_ensemble_ranges(ensemble_tuples)

    groups = []
    for ensemble_range in ensemble_ranges:
        txt = ''
        for name, value in zip('ripf', ensemble_range):
            txt += name
            if value[0] == value[1]:
                txt += f"{value[0]}"
            else:
                txt += f"({value[0]}:{value[1]})"
        groups.append(txt)

    return groups


def _create_ensemble_ranges(
    ensembles: Sequence[tuple[int,
                              ...]], ) -> list[tuple[tuple[int, int], ...]]:
    """Create ranges from tuples.

    Examples
    --------
    Input ensemble member tuple (1, 1, 1) represents 'r1i1p1'.
    The input tuples will be converted to ranges, for example
    ensembles=[
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (1, 1, 2),
    ]
    will return [
        ((1, 3), (1, 1), (1, 1)),
        ((1, 1), (1, 1), (2, 2)),
    ].
    """

    def order(i, ens):
        prefix, suffix = ens[:i], ens[i + 1:]
        return (prefix, suffix, ens[i])

    def grouper(i, ens):
        prefix, suffix = ens[:i], ens[i + 1:]
        return (prefix, suffix)

    for i in range(len(ensembles[0])):
        grouped_ensembles = []
        ensembles = sorted(ensembles, key=partial(order, i))
        for (prefix,
             suffix), ibunch in itertools.groupby(ensembles,
                                                  key=partial(grouper, i)):
            bunch = list(ibunch)
            prev = bunch[0][i]
            groups = [[prev]]
            for ensemble in bunch[1:]:
                if ensemble[i] == prev + 1:
                    prev += 1
                else:
                    groups[-1].append(prev)
                    prev = ensemble[i]
                    groups.append([prev])
            groups[-1].append(prev)
            result = []
            for group in groups:
                item = prefix + (tuple(group), ) + suffix
                result.append(item)
            grouped_ensembles.extend(result)

        ensembles = grouped_ensembles

    return sorted(ensembles)  # type: ignore


def _clean_recipe(recipe: Recipe, diagnostics: list[str]) -> Recipe:
    """Clean up the input recipe."""
    # Format description nicer
    if 'documentation' in recipe:
        doc = recipe['documentation']
        for key in ['title', 'description']:
            if key in doc:
                doc[key] = doc[key].strip()

    # Filter out unused diagnostics
    recipe['diagnostics'] = {
        k: v
        for k, v in recipe['diagnostics'].items() if k in diagnostics
    }

    # Remove legacy supplementary definitions form the recipe
    nested_delete(
        recipe.get('preprocessors', {}),
        'fx_variables',
        in_place=True,
    )

    return recipe


def datasets_to_recipe(
    datasets: Iterable[Dataset],
    recipe: Path | str | dict[str, Any] | None = None,
) -> dict:
    """Create or update a recipe from datasets.

    Parameters
    ----------
    datasets
        Datasets to use in the recipe.
    recipe
        :ref:`Recipe <recipe>` to load the datasets from. The value
        provided here should be either a path to a file, a recipe file
        that has been loaded using e.g. :func:`yaml.safe_load`, or an
        :obj:`str` that can be loaded using :func:`yaml.safe_load`.

    Examples
    --------
    See :ref:`/notebooks/composing-recipes.ipynb` for example use cases.

    Returns
    -------
    dict
        The recipe with the datasets. To convert the :obj:`dict` to a
        :ref:`recipe <recipe>`, use e.g. :func:`yaml.safe_dump`.

    Raises
    ------
    RecipeError
        Raised when a dataset is missing the ``diagnostic`` facet.
    """
    recipe = _load_recipe(recipe)
    dataset_recipe = _datasets_to_recipe(datasets)
    _clean_recipe(recipe, diagnostics=dataset_recipe['diagnostics'])

    # Remove dataset sections from recipe
    recipe.pop('datasets', None)
    nested_delete(recipe, 'additional_datasets', in_place=True)

    # Update datasets section
    if 'datasets' in dataset_recipe:
        recipe['datasets'] = dataset_recipe['datasets']

    for diag, dataset_diagnostic in dataset_recipe['diagnostics'].items():
        if diag not in recipe['diagnostics']:
            recipe['diagnostics'][diag] = {}
        diagnostic = recipe['diagnostics'][diag]
        # Update diagnostic level datasets
        if 'additional_datasets' in dataset_diagnostic:
            additional_datasets = dataset_diagnostic['additional_datasets']
            diagnostic['additional_datasets'] = additional_datasets
        # Update variable level datasets
        if 'variables' in dataset_diagnostic:
            diagnostic['variables'] = dataset_diagnostic['variables']

    return recipe
