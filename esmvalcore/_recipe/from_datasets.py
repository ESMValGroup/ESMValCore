"""Functions for creating/updating a recipe with `Dataset`s"""
from __future__ import annotations

import itertools
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from nested_lookup import nested_delete

if TYPE_CHECKING:
    from esmvalcore.dataset import Dataset

from esmvalcore.exceptions import RecipeError

logger = logging.getLogger(__name__)

Recipe = dict[str, Any]
Facets = dict[str, Any]


def _datasets_to_raw_recipe(datasets: Iterable[Dataset]) -> Recipe:
    """Convert datasets to a recipe dict."""
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

    recipe = {'diagnostics': diagnostics}
    return recipe


def _datasets_to_recipe(datasets: Iterable[Dataset]) -> Recipe:
    """Convert datasets to a condensed recipe dict."""
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

    # TODO: make recipe look nicer
    # - deduplicate by moving datasets up from variable to diagnostic to recipe

    return recipe


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
    """This is the inverse operation of `Dataset.from_ranges` for ensembles."""
    def grouper(facets):
        return tuple((k, facets[k]) for k in sorted(facets) if k != 'ensemble')

    result = []
    for group_facets, group in itertools.groupby(dataset_facets, key=grouper):
        ensembles = [f['ensemble'] for f in group if 'ensemble' in f]
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
    n_items = len(ensembles[0])
    for i in range(n_items):

        def order(ens):
            prefix, suffix = ens[:i], ens[i + 1:]
            return (prefix, suffix, ens[i])

        def grouper(ens):
            prefix, suffix = ens[:i], ens[i + 1:]
            return (prefix, suffix)

        grouped_ensembles = []
        ensembles = sorted(ensembles, key=order)
        for (prefix, suffix), ibunch in itertools.groupby(ensembles,
                                                          key=grouper):
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


def datasets_to_recipe(
    datasets: Iterable[Dataset],
    recipe: Recipe | None = None,
) -> Recipe:
    """Create or update a recipe from datasets.

    Parameters
    ----------
    datasets
        Datasets to use in the recipe.
    recipe
        If provided, the datasets in the recipe will be replaced.

    Returns
    -------
    dict
        The recipe with the datasets.

    Raises
    ------
    RecipeError
        Raised when a dataset is missing the ``diagnostic`` facet.
    """
    # TODO: should recipe be a dict, a string, or a file?
    if recipe is None:
        recipe = {
            'diagnostics': {},
        }
    else:
        recipe = deepcopy(recipe)

    # Remove dataset sections from recipe
    recipe.pop('datasets', None)
    nested_delete(recipe, 'additional_datasets', in_place=True)

    # Update datasets section
    dataset_recipe = _datasets_to_recipe(datasets)
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

    # Format description nicer
    if 'documentation' in recipe:
        doc = recipe['documentation']
        if 'description' in doc:
            doc['description'] = doc['description'].strip()

    return recipe
