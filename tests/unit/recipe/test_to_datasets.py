import textwrap
from pathlib import Path

import pytest
import yaml

from esmvalcore._recipe import to_datasets
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError
from esmvalcore.local import LocalFile


def test_from_recipe(session):
    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: cccma_cgcm3_1
        ensemble: run1
        exp: historical
        frequency: mon
        mip: A1
        project: CMIP3
      - dataset: EC-EARTH
        ensemble: r1i1p1
        exp: historical
        mip: Amon
        project: CMIP5
      - dataset: AWI-ESM-1-1-LR
        ensemble: r1i1p1f1
        exp: historical
        grid: gn
        mip: Amon
        project: CMIP6
      - dataset: RACMO22E
        driver: MOHC-HadGEM2-ES
        domain: EUR-11
        ensemble: r1i1p1
        exp: historical
        mip: mon
        project: CORDEX
      - dataset: CERES-EBAF
        mip: Amon
        project: obs4MIPs

    preprocessors:
      preprocessor1:
        extract_levels:
          levels: 85000
          scheme: nearest

    diagnostics:
      diagnostic1:
        variables:
          ta850:
            short_name: ta
            preprocessor: preprocessor1
    """)
    datasets = Dataset.from_recipe(recipe_txt, session)

    reference = [
        Dataset(
            alias='CMIP3',
            dataset='cccma_cgcm3_1',
            diagnostic='diagnostic1',
            ensemble='run1',
            exp='historical',
            frequency='mon',
            mip='A1',
            preprocessor='preprocessor1',
            project='CMIP3',
            recipe_dataset_index=0,
            short_name='ta',
            variable_group='ta850',
        ),
        Dataset(
            alias='CMIP5',
            dataset='EC-EARTH',
            diagnostic='diagnostic1',
            ensemble='r1i1p1',
            exp='historical',
            mip='Amon',
            preprocessor='preprocessor1',
            project='CMIP5',
            recipe_dataset_index=1,
            short_name='ta',
            variable_group='ta850',
        ),
        Dataset(
            alias='CMIP6',
            dataset='AWI-ESM-1-1-LR',
            diagnostic='diagnostic1',
            ensemble='r1i1p1f1',
            exp='historical',
            grid='gn',
            mip='Amon',
            preprocessor='preprocessor1',
            project='CMIP6',
            recipe_dataset_index=2,
            short_name='ta',
            variable_group='ta850',
        ),
        Dataset(
            alias='CORDEX',
            dataset='RACMO22E',
            diagnostic='diagnostic1',
            driver='MOHC-HadGEM2-ES',
            domain='EUR-11',
            ensemble='r1i1p1',
            exp='historical',
            mip='mon',
            preprocessor='preprocessor1',
            project='CORDEX',
            recipe_dataset_index=3,
            short_name='ta',
            variable_group='ta850',
        ),
        Dataset(
            alias='obs4MIPs',
            dataset='CERES-EBAF',
            diagnostic='diagnostic1',
            mip='Amon',
            preprocessor='preprocessor1',
            project='obs4MIPs',
            recipe_dataset_index=4,
            short_name='ta',
            variable_group='ta850',
        ),
    ]
    for ref_ds in reference:
        ref_ds.session = session

    assert datasets == reference


@pytest.mark.parametrize('path_type', [str, Path])
def test_from_recipe_file(tmp_path, session, path_type):
    recipe_file = tmp_path / 'recipe_test.yml'
    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: AWI-ESM-1-1-LR
        grid: gn

    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            project: CMIP6

    """)
    recipe_file.write_text(recipe_txt, encoding='utf-8')
    datasets = Dataset.from_recipe(
        path_type(recipe_file),
        session,
    )
    assert len(datasets) == 1


def test_from_recipe_dict(session):
    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: AWI-ESM-1-1-LR
        grid: gn

    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            project: CMIP6

    """)
    recipe_dict = yaml.safe_load(recipe_txt)
    datasets = Dataset.from_recipe(recipe_dict, session)
    assert len(datasets) == 1


def test_merge_supplementaries_dataset_takes_priority(session):
    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: AWI-ESM-1-1-LR
        grid: gn
      - dataset: BCC-ESM1
        grid: gn
        supplementary_variables:
            - short_name: areacella
              exp: 1pctCO2

    preprocessors:
      global_mean:
        area_statistics:
          statistic: mean

    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            preprocessor: global_mean_land
            project: CMIP6
            supplementary_variables:
              - short_name: areacella
                mip: fx

    """)

    datasets = Dataset.from_recipe(recipe_txt, session)
    print(datasets)
    assert len(datasets) == 2
    assert all(len(ds.supplementaries) == 1 for ds in datasets)
    assert datasets[0].supplementaries[0].facets['exp'] == 'historical'
    assert datasets[1].supplementaries[0].facets['exp'] == '1pctCO2'


def test_merge_supplementaries_combine_dataset_with_variable(session):
    recipe_txt = textwrap.dedent("""
    datasets:
      - dataset: AWI-ESM-1-1-LR
        grid: gn
        supplementary_variables:
          - short_name: sftlf
            mip: fx

    preprocessors:
      global_mean_land:
        mask_landsea:
          mask_out: sea
        area_statistics:
          statistic: mean

    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            preprocessor: global_mean_land
            project: CMIP6
            supplementary_variables:
              - short_name: areacella
                mip: fx

    """)

    datasets = Dataset.from_recipe(recipe_txt, session)
    print(datasets)
    assert len(datasets) == 1
    assert len(datasets[0].supplementaries) == 2
    assert datasets[0].supplementaries[0].facets['short_name'] == 'areacella'
    assert datasets[0].supplementaries[1].facets['short_name'] == 'sftlf'


def test_merge_supplementaries_missing_short_name_fails(session):
    recipe_txt = textwrap.dedent("""
    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            project: CMIP6
            supplementary_variables:
              - mip: fx
            additional_datasets:
              - dataset: AWI-ESM-1-1-LR
                grid: gn
    """)

    with pytest.raises(RecipeError):
        Dataset.from_recipe(recipe_txt, session)


def test_get_input_datasets_derive(session):
    dataset = Dataset(
        dataset='ERA5',
        project='native6',
        mip='E1hr',
        short_name='rlus',
        alias='ERA5',
        derive=True,
        force_derivation=True,
        frequency='1hr',
        recipe_dataset_index=0,
        tier='3',
        type='reanaly',
        version='v1',
    )
    rlds, rlns = to_datasets._get_input_datasets(dataset)
    assert rlds['short_name'] == 'rlds'
    assert rlds['long_name'] == 'Surface Downwelling Longwave Radiation'
    assert rlds['frequency'] == '1hr'
    assert rlns['short_name'] == 'rlns'
    assert rlns['long_name'] == 'Surface Net downward Longwave Radiation'
    assert rlns['frequency'] == '1hr'


def test_max_years(session):
    recipe_txt = textwrap.dedent("""
    diagnostics:
      diagnostic1:
        variables:
          tas:
            ensemble: r1i1p1f1
            exp: historical
            mip: Amon
            project: CMIP6
            start_year: 2000
            end_year: 2010
            additional_datasets:
              - dataset: AWI-ESM-1-1-LR
                grid: gn
    """)
    session['max_years'] = 2
    datasets = Dataset.from_recipe(recipe_txt, session)
    assert datasets[0].facets['timerange'] == '2000/2001'


@pytest.mark.parametrize('found_files', [True, False])
def test_dataset_from_files_fails(monkeypatch, found_files):

    def from_files(_):
        file = LocalFile('/path/to/file')
        file.facets = {'facets1': 'value1'}
        dataset = Dataset(
            dataset='*',
            short_name='tas',
        )
        dataset.files = [file] if found_files else []
        dataset._file_globs = ['/path/to/tas_*.nc']
        return [dataset]

    monkeypatch.setattr(Dataset, 'from_files', from_files)

    dataset = Dataset(
        dataset='*',
        short_name='tas',
    )

    with pytest.raises(RecipeError, match="Unable to replace dataset.*"):
        to_datasets._dataset_from_files(dataset)


def test_fix_cmip5_fx_ensemble(monkeypatch):

    def find_files(self):
        if self.facets['ensemble'] == 'r0i0p0':
            self._files = ['file1.nc']

    monkeypatch.setattr(Dataset, 'find_files', find_files)

    dataset = Dataset(
        dataset='dataset1',
        short_name='orog',
        mip='fx',
        project='CMIP5',
        ensemble='r1i1p1',
    )

    to_datasets._fix_cmip5_fx_ensemble(dataset)

    assert dataset['ensemble'] == 'r0i0p0'


def test_get_supplementary_short_names(monkeypatch):

    def _update_cmor_facets(facets):
        facets['modeling_realm'] = 'atmos'

    monkeypatch.setattr(
        to_datasets,
        '_update_cmor_facets',
        _update_cmor_facets,
    )
    facets = {
        'short_name': 'tas',
    }
    result = to_datasets._get_supplementary_short_names(facets, 'mask_landsea')
    assert result == ['sftlf']


def test_append_missing_supplementaries():
    supplementaries = [
        {
            'short_name': 'areacella',
        },
    ]
    facets = {
        'short_name': 'tas',
        'project': 'CMIP6',
        'mip': 'Amon',
    }

    settings = {
        'mask_landsea': {
            'mask_out': 'land'
        },
        'area_statistics': {
            'operator': 'mean'
        },
    }

    to_datasets._append_missing_supplementaries(supplementaries, facets,
                                                settings)

    short_names = {f['short_name'] for f in supplementaries}
    assert short_names == {'areacella', 'sftlf'}


def test_report_unexpanded_globs(mocker):
    dataset = Dataset(
        alias='CMIP5',
        dataset='*',
        diagnostic='diagnostic1',
        ensemble='r1i1p1',
        exp='historical',
        mip='Amon',
        preprocessor='preprocessor1',
        project='CMIP5',
        recipe_dataset_index=1,
        short_name='ta',
        variable_group='ta850',
    )
    file = mocker.Mock(facets={'dataset': '*'})
    dataset.files = [file]
    unexpanded_globs = {'dataset': '*'}

    msg = to_datasets._report_unexpanded_globs(
        dataset, dataset, unexpanded_globs
    )

    assert 'paths to the' not in msg
