import textwrap

import pytest
import yaml

from esmvalcore._recipe.from_datasets import (
    _group_ensemble_members,
    _group_ensemble_names,
    _group_identical_facets,
    _move_one_level_up,
    _to_frozen,
    datasets_to_recipe,
)
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError


def test_to_frozen():
    data = {
        'abc': 'x',
        'a': {
            'b': [
                'd',
                'c',
            ],
        },
    }

    result = _to_frozen(data)
    expected = (
        (
            'a',
            ((
                'b',
                (
                    'c',
                    'd',
                ),
            ), ),
        ),
        ('abc', 'x'),
    )

    assert result == expected


def test_datasets_to_recipe():
    dataset = Dataset(
        short_name='tas',
        dataset='dataset1',
        diagnostic='diagnostic1',
    )
    datasets = [dataset]
    datasets.append(dataset.copy(short_name='pr'))
    datasets.append(dataset.copy(dataset='dataset2'))
    datasets.append(dataset.copy(diagnostic='diagnostic2'))
    datasets.append(dataset.copy(diagnostic='diagnostic2', dataset='dataset3'))

    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: 'dataset1'
    diagnostics:
      diagnostic1:
        variables:
          tas:
            additional_datasets:
              - dataset: 'dataset2'
          pr: {}
      diagnostic2:
        variables:
          tas: {}
        additional_datasets:
          - dataset: 'dataset3'
    """)
    recipe = yaml.safe_load(recipe_txt)

    assert datasets_to_recipe(datasets) == recipe


def test_update_datasets_in_recipe():
    existing_recipe_txt = textwrap.dedent("""
    documentation:
      description: |
        This recipe computes something interesting.
    datasets:
      - {dataset: 'dataset1'}
    diagnostics:
      diagnostic1:
        variables:
          ta: {}
    """)
    existing_recipe = yaml.safe_load(existing_recipe_txt)

    dataset = Dataset(
        short_name='ta',
        dataset='dataset2',
        diagnostic='diagnostic1',
    )

    recipe_txt = textwrap.dedent("""
    documentation:
      description: This recipe computes something interesting.
    datasets:
      - {dataset: 'dataset2'}
    diagnostics:
      diagnostic1:
        variables:
          ta: {}
    """)
    recipe = yaml.safe_load(recipe_txt)

    assert datasets_to_recipe([dataset], recipe=existing_recipe) == recipe


def test_supplementary_datasets_to_recipe():
    dataset = Dataset(
        short_name='ta',
        dataset='dataset1',
    )
    dataset['diagnostic'] = 'diagnostic1'
    dataset['variable_group'] = 'group1'
    dataset.add_supplementary(short_name='areacella')

    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: 'dataset1'
    diagnostics:
      diagnostic1:
        variables:
          group1:
            short_name: 'ta'
            supplementary_variables:
              - short_name: areacella
    """)
    recipe = yaml.safe_load(recipe_txt)
    assert datasets_to_recipe([dataset]) == recipe


def test_datasets_to_recipe_group_ensembles():
    datasets = [
        Dataset(
            short_name='ta',
            ensemble='r1i1p1f1',
            dataset='dataset1',
        ),
        Dataset(
            short_name='ta',
            ensemble='r2i1p1f1',
            dataset='dataset1',
        ),
        Dataset(
            short_name='ta',
            dataset='dataset2',
        ),
    ]
    for dataset in datasets:
        dataset.facets['diagnostic'] = 'diagnostic1'
    recipe_txt = textwrap.dedent("""
    datasets:
        - {dataset: 'dataset1', ensemble: 'r(1:2)i1p1f1'}
        - {dataset: 'dataset2'}
    diagnostics:
      diagnostic1:
        variables:
          ta: {}
    """)
    recipe = yaml.safe_load(recipe_txt)

    assert datasets_to_recipe(datasets) == recipe


def test_datasets_to_recipe_no_diagnostic():
    dataset = Dataset(short_name='tas')
    msg = "'diagnostic' facet missing from .*"
    with pytest.raises(RecipeError, match=msg):
        datasets_to_recipe([dataset])


def test_group_identical_facets():
    variable = {
        'short_name':
        'tas',
        'additional_datasets': [
            {
                'dataset': 'dataset1',
                'ensemble': 'r1i1p1f1',
            },
            {
                'dataset': 'dataset2',
                'ensemble': 'r1i1p1f1',
            },
        ],
    }

    result = _group_identical_facets(variable)

    expected = {
        'short_name':
        'tas',
        'ensemble':
        'r1i1p1f1',
        'additional_datasets': [
            {
                'dataset': 'dataset1',
            },
            {
                'dataset': 'dataset2',
            },
        ],
    }

    assert result == expected


def test_group_ensemble_members():
    datasets = [
        Dataset(
            dataset='dataset1',
            ensemble='r1i1p1f1',
            grid='gn',
        ),
        Dataset(
            dataset='dataset1',
            ensemble='r1i1p1f1',
            grid='gr1',
        ),
        Dataset(
            dataset='dataset1',
            ensemble='r2i1p1f1',
            grid='gn',
        ),
    ]
    result = _group_ensemble_members(ds.facets for ds in datasets)
    print(result)
    assert result == [
        {
            'dataset': 'dataset1',
            'ensemble': 'r(1:2)i1p1f1',
            'grid': 'gn',
        },
        {
            'dataset': 'dataset1',
            'ensemble': 'r1i1p1f1',
            'grid': 'gr1',
        },
    ]


def test_group_ensemble_members_mix_of_versions():
    datasets = [
        Dataset(
            dataset='dataset1',
            ensemble='r1i1p1f1',
            exp=['historical', 'ssp585'],
            version='v1',
        ),
        Dataset(
            dataset='dataset1',
            ensemble='r2i1p1f1',
            exp=['historical', 'ssp585'],
            version='v1',
        ),
        Dataset(
            dataset='dataset1',
            ensemble='r3i1p1f1',
            exp=['historical', 'ssp585'],
            version=['v1', 'v2'],
        ),
    ]
    result = _group_ensemble_members(ds.facets for ds in datasets)
    print(result)
    assert result == [
        {
            'dataset': 'dataset1',
            'ensemble': 'r3i1p1f1',
            'exp': ['historical', 'ssp585'],
            'version': ['v1', 'v2'],
        },
        {
            'dataset': 'dataset1',
            'ensemble': 'r(1:2)i1p1f1',
            'exp': ['historical', 'ssp585'],
            'version': 'v1',
        },
    ]


def test_group_ensembles_cmip5():
    ensembles = [
        "r1i1p1",
        "r2i1p1",
        "r3i1p1",
        "r4i1p1",
        "r1i2p1",
    ]
    groups = _group_ensemble_names(ensembles)
    expected = ['r1i2p1', 'r(1:4)i1p1']
    print(groups)
    print(expected)
    assert groups == expected


def test_group_ensembles_cmip6():
    ensembles = [
        "r1i1p1f1",
        "r4i1p1f1",
        "r3i1p2f1",
        "r4i1p2f1",
        "r3i1p1f1",
    ]
    groups = _group_ensemble_names(ensembles)
    expected = ['r1i1p1f1', 'r(3:4)i1p(1:2)f1']
    print(groups)
    print(expected)
    assert groups == expected


def test_move_one_level_up_diagnostic():
    diagnostic = {
        'variables': {
            'tas': {
                'mip':
                'Amon',
                'additional_datasets': [
                    {
                        'dataset': 'dataset1'
                    },
                    {
                        'dataset': 'dataset2'
                    },
                ],
            },
            'pr': {
                'additional_datasets': [
                    {
                        'dataset': 'dataset1'
                    },
                ],
            },
        },
    }

    _move_one_level_up(diagnostic, 'variables', 'additional_datasets')

    expected = {
        'variables': {
            'tas': {
                'mip': 'Amon',
                'additional_datasets': [
                    {
                        'dataset': 'dataset2'
                    },
                ],
            },
            'pr': {},
        },
        'additional_datasets': [
            {
                'dataset': 'dataset1'
            },
        ],
    }

    assert diagnostic == expected


def test_move_one_level_up_recipe():
    recipe = {
        'diagnostics': {
            'diagnostic1': {
                'variables': {
                    'tas': {
                        'mip': 'Amon',
                    },
                },
                'additional_datasets': [
                    {
                        'dataset': 'dataset1'
                    },
                    {
                        'dataset': 'dataset2'
                    },
                ],
            },
        },
    }

    _move_one_level_up(recipe, 'diagnostics', 'datasets')

    expected = {
        'datasets': [
            {
                'dataset': 'dataset1'
            },
            {
                'dataset': 'dataset2'
            },
        ],
        'diagnostics': {
            'diagnostic1': {
                'variables': {
                    'tas': {
                        'mip': 'Amon',
                    },
                },
            },
        },
    }

    assert recipe == expected
