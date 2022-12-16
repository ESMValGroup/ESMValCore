import textwrap

import yaml

from esmvalcore._recipe.datasets import (
    datasets_to_recipe,
    group_identical_facets,
)
from esmvalcore.dataset import Dataset


def test_datasets_to_recipe():
    datasets = [
        Dataset(
            short_name='ta',
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

    diagnostics:
      diagnostic1:
        variables:
          ta:
            additional_datasets:
              - {dataset: 'dataset1'}
              - {dataset: 'dataset2'}

    """)
    recipe = yaml.safe_load(recipe_txt)

    assert datasets_to_recipe(datasets) == recipe


def test_ancillary_datasets_to_recipe():
    dataset = Dataset(
        short_name='ta',
        dataset='dataset1',
    )
    dataset['diagnostic'] = 'diagnostic1'
    dataset['variable_group'] = 'group1'
    dataset.add_ancillary(short_name='areacella')

    recipe_txt = textwrap.dedent("""

    diagnostics:
      diagnostic1:
        variables:
          group1:
            short_name: 'ta'
            ancillary_variables:
              - short_name: areacella
            additional_datasets:
              - dataset: 'dataset1'

    """)
    recipe = yaml.safe_load(recipe_txt)
    assert datasets_to_recipe([dataset]) == recipe


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

    result = group_identical_facets(variable)

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
