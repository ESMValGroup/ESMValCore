import pprint
import textwrap

import pytest
import yaml

from esmvalcore.dataset import (
    Dataset,
    datasets_from_recipe,
    datasets_to_recipe,
)
from esmvalcore.exceptions import RecipeError
from esmvalcore.experimental import CFG


@pytest.fixture
def session(tmp_path):
    CFG['output_dir'] = tmp_path
    return CFG.start_session('recipe_test')


def test_repr():

    ds = Dataset(short_name='tas', dataset='dataset1')

    assert repr(ds) == "Dataset(dataset='dataset1', short_name='tas')"


def test_repr_ancillary():
    ds = Dataset(dataset='dataset1', short_name='tas')
    ds.add_ancillary(short_name='areacella')

    assert repr(ds) == textwrap.dedent("""
        Dataset(dataset='dataset1', short_name='tas')
        .add_ancillary(dataset='dataset1', short_name='areacella')
        """).strip()


def test_datsets_from_recipe(session):

    recipe_txt = textwrap.dedent("""

    diagnostics:
      diagnostic1:
        variables:
          tas:
            project: CMIP5
            mip: Amon
            additional_datasets:
              - {dataset: dataset1}
    """)
    recipe = yaml.safe_load(recipe_txt)

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tas',
        short_name='tas',
        dataset='dataset1',
        project='CMIP5',
        mip='Amon',
        preprocessor='default',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.session = session

    print(datasets_from_recipe(recipe, session))
    print([dataset])
    assert datasets_from_recipe(recipe, session) == [dataset]


def test_datasets_from_complicated_recipe(session):

    recipe_txt = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', project: CMIP6}

    diagnostics:
      diagnostic1:
        additional_datasets:
          - {dataset: 'dataset2', project: CMIP6}
        variables:
          ta:
            mip: Amon
          pr:
            mip: Amon
            additional_datasets:
              - {dataset: 'dataset3', project: CMIP5}
      diagnostic2:
        variables:
          tos:
            mip: Omon
    """)

    recipe = yaml.safe_load(recipe_txt)

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='CMIP6_dataset1',
            recipe_dataset_index=0,
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset2',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='CMIP6_dataset2',
            recipe_dataset_index=1,
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset1',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='CMIP6_dataset1',
            recipe_dataset_index=0,
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset2',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='CMIP6_dataset2',
            recipe_dataset_index=1,
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='pr',
            short_name='pr',
            dataset='dataset3',
            project='CMIP5',
            mip='Amon',
            preprocessor='default',
            alias='CMIP5',
            recipe_dataset_index=2,
        ),
        Dataset(
            diagnostic='diagnostic2',
            variable_group='tos',
            short_name='tos',
            dataset='dataset1',
            project='CMIP6',
            mip='Omon',
            preprocessor='default',
            alias='dataset1',
            recipe_dataset_index=0,
        ),
    ]

    assert datasets_from_recipe(recipe, session) == datasets


def test_expand_datasets_from_recipe(session):

    recipe_txt = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', ensemble: r(1:2)i1p1}

    diagnostics:
      diagnostic1:
        variables:
          ta:
            mip: Amon
            project: CMIP6
    """)
    recipe = yaml.safe_load(recipe_txt)

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            ensemble='r1i1p1',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='r1i1p1',
            recipe_dataset_index=0,
        ),
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            ensemble='r2i1p1',
            project='CMIP6',
            mip='Amon',
            preprocessor='default',
            alias='r2i1p1',
            recipe_dataset_index=1,
        ),
    ]
    for dataset in datasets:
        dataset.session = session

    assert datasets_from_recipe(recipe, session) == datasets


def test_ancillary_datasets_from_recipe(session):

    recipe_txt = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', ensemble: r1i1p1}

    diagnostics:
      diagnostic1:
        variables:
          tos:
            project: CMIP5
            mip: Omon
            ancillary_variables:
              - short_name: sftof
                mip: fx
    """)
    recipe = yaml.safe_load(recipe_txt)

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tos',
        short_name='tos',
        dataset='dataset1',
        ensemble='r1i1p1',
        project='CMIP5',
        mip='Omon',
        preprocessor='default',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.ancillaries = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='tos',
            short_name='sftof',
            dataset='dataset1',
            ensemble='r1i1p1',
            project='CMIP5',
            mip='fx',
        ),
    ]
    dataset.session = session

    print(datasets_from_recipe(recipe, session))
    print([dataset])
    assert datasets_from_recipe(recipe, session) == [dataset]


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
            additional_datasets:
              - dataset: 'dataset1'
                short_name: 'ta'
                ancillary_variables:
                  - short_name: areacella

    """)
    recipe = yaml.safe_load(recipe_txt)
    assert datasets_to_recipe([dataset]) == recipe


def test_expand_ensemble():

    dataset = Dataset(ensemble='r(1:2)i(2:3)p(3:4)')

    expanded = dataset._expand_range('ensemble')

    ensembles = [
        'r1i2p3',
        'r1i2p4',
        'r1i3p3',
        'r1i3p4',
        'r2i2p3',
        'r2i2p4',
        'r2i3p3',
        'r2i3p4',
    ]
    assert expanded == ensembles


def test_expand_subexperiment():

    dataset = Dataset(sub_experiment='s(1998:2005)')

    expanded = dataset._expand_range('sub_experiment')

    subexperiments = [
        's1998',
        's1999',
        's2000',
        's2001',
        's2002',
        's2003',
        's2004',
        's2005',
    ]

    assert expanded == subexperiments


def test_expand_ensemble_nolist():

    dataset = Dataset(
        dataset='XYZ',
        ensemble=['r1i1p1', 'r(1:2)i1p1'],
    )

    with pytest.raises(RecipeError):
        dataset._expand_range('ensemble')
