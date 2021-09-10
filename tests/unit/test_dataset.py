import textwrap

from esmvalcore._dataset import Dataset, datasets_from_recipe


def test_datasets_from_recipe(tmp_path):

    recipe = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', project: 'project1'}

    diagnostics:
      diagnostic1:
        additional_datasets:
          - {dataset: 'dataset2', project: 'project2'}
        variables:
          ta:
          pr:
            additional_datasets:
              - {dataset: 'dataset3', project: 'project1'}
      diagnostic2:
        variables:
          tos:
    """)
    recipe_file = tmp_path / 'recipe_test.yml'
    recipe_file.write_text(recipe)

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            project='project1',
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset2',
            project='project2',
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset1',
            project='project1',
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset2',
            project='project2',
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset3',
            project='project1',
        ),
        Dataset(
            diagnostic='diagnostic2',
            variable_group='tos',
            short_name='tos',
            dataset='dataset1',
            project='project1',
        ),
    ]

    assert datasets_from_recipe(recipe_file) == datasets


def test_expand_datasets_from_recipe(tmp_path):

    recipe = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', project: 'project1', ensemble: r(1:2)i1p1}

    diagnostics:
      diagnostic1:
        variables:
          ta:
    """)
    recipe_file = tmp_path / 'recipe_test.yml'
    recipe_file.write_text(recipe)

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            project='project1',
            ensemble='r1i1p1',
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            project='project1',
            ensemble='r2i1p1',
        ),
    ]

    assert datasets_from_recipe(recipe_file) == datasets
