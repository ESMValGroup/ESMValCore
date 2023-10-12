import textwrap
from collections import defaultdict
from pathlib import Path
from unittest import mock

import pyesgf
import pytest

import esmvalcore.dataset
import esmvalcore.local
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset
from esmvalcore.esgf import ESGFFile
from esmvalcore.exceptions import InputFilesNotFound, RecipeError


def test_repr():
    ds = Dataset(short_name='tas', dataset='dataset1')

    assert repr(ds) == textwrap.dedent("""
        Dataset:
        {'dataset': 'dataset1', 'short_name': 'tas'}
    """).strip()


def test_repr_session(mocker):
    ds = Dataset(short_name='tas', dataset='dataset1')
    ds.session = mocker.Mock()
    ds.session.session_name = 'test-session'
    assert repr(ds) == textwrap.dedent("""
        Dataset:
        {'dataset': 'dataset1', 'short_name': 'tas'}
        session: 'test-session'
    """).strip()


def test_repr_supplementary():
    ds = Dataset(dataset='dataset1', short_name='tas')
    ds.add_supplementary(short_name='areacella')

    assert repr(ds) == textwrap.dedent("""
        Dataset:
        {'dataset': 'dataset1', 'short_name': 'tas'}
        supplementaries:
          {'dataset': 'dataset1', 'short_name': 'areacella'}
        """).strip()


@pytest.mark.parametrize(
    "separator,join_lists,output",
    [
        ('_', False, "1_d_dom_a_('e1', 'e2')_['ens2', 'ens1']_g1_v1"),
        ('_', True, "1_d_dom_a_e1-e2_ens2-ens1_g1_v1"),
        (' ', False, "1 d dom a ('e1', 'e2') ['ens2', 'ens1'] g1 v1"),
        (' ', True, "1 d dom a e1-e2 ens2-ens1 g1 v1"),
    ]
)
def test_get_joined_summary_facet(separator, join_lists, output):
    ds = Dataset(
        test='this should not appear',
        rcm_version='1',
        driver='d',
        domain='dom',
        activity='a',
        exp=('e1', 'e2'),
        ensemble=['ens2', 'ens1'],
        grid='g1',
        version='v1',
    )
    joined_str = ds._get_joined_summary_facets(
        separator, join_lists=join_lists
    )
    assert joined_str == output


def test_short_summary():
    ds = Dataset(
        project='CMIP6',
        dataset='dataset1',
        short_name='tos',
        mip='Omon',
    )
    ds.add_supplementary(short_name='areacello', mip='Ofx')
    ds.add_supplementary(short_name='volcello')
    expected = ("Dataset: tos, Omon, CMIP6, dataset1, "
                "supplementaries: areacello, Ofx; volcello")
    assert ds.summary(shorten=True) == expected


def test_long_summary():
    ds = Dataset(dataset='dataset1', short_name='tas')
    assert ds.summary(shorten=False) == repr(ds)


def test_session_setter():
    ds = Dataset(short_name='tas')
    ds.add_supplementary(short_name='areacella')
    assert ds._session is None
    assert ds.supplementaries[0]._session is None

    ds.session

    assert isinstance(ds.session, esmvalcore.config.Session)
    assert ds.session == ds.supplementaries[0].session


@pytest.mark.parametrize(
    'facets,added_facets',
    [
        [
            {
                'short_name': 'areacella',
                'project': 'ICON',
                'mip': 'fx',
                'dataset': 'ICON',
            },
            {
                # Added from CMOR table
                'original_short_name': 'areacella',
                'standard_name': 'cell_area',
                'long_name': 'Grid-Cell Area for Atmospheric Grid Variables',
                'units': 'm2',
                'modeling_realm': ['atmos', 'land'],
                'frequency': 'fx',
                # Added from extra facets YAML file
                'latitude': 'grid_latitude',
                'longitude': 'grid_longitude',
                'raw_name': 'cell_area',
            },
        ],
        [
            {
                'short_name': 'zg',
                'mip': 'A1',
                'project': 'CMIP3',
                'dataset': 'bccr_bcm2_0',
                'frequency': 'mon',
                'exp': 'historical',
                'ensemble': 'r1i1p1',
                'modeling_realm': 'atmos',
            },
            {
                # Added from CMOR table
                'original_short_name': 'zg',
                'long_name': 'Geopotential Height',
                'standard_name': 'geopotential_height',
                'units': 'm',
                # Added from extra facets YAML file
                'institute': ['BCCR'],
            },
        ],
        [
            {
                'short_name': 'pr',
                'mip': '3hr',
                'project': 'CMIP5',
                'dataset': 'CNRM-CM5',
                'exp': 'historical',
                'ensemble': 'r1i1p1',
                'timerange': '2000/2000',
            },
            {
                # Added from CMOR table
                'original_short_name': 'pr',
                'frequency': '3hr',
                'long_name': 'Precipitation',
                'modeling_realm': ['atmos'],
                'standard_name': 'precipitation_flux',
                'units': 'kg m-2 s-1',
                # Added from extra facets YAML file
                'institute': ['CNRM-CERFACS'],
                'product': ['output1', 'output2'],
            },
        ],
        [
            {
                'short_name': 'pr',
                'mip': '3hr',
                'project': 'CMIP6',
                'dataset': 'HadGEM3-GC31-LL',
                'exp': 'historical',
                'ensemble': 'r2i1p1f1',
                'grid': 'gn',
                'timerange': '2000/2001',
            },
            {
                # Added from CMOR table
                'activity': 'CMIP',
                'frequency': '3hr',
                'institute': ['MOHC', 'NERC'],
                'long_name': 'Precipitation',
                'modeling_realm': ['atmos'],
                'original_short_name': 'pr',
                'standard_name': 'precipitation_flux',
                'timerange': '2000/2001',
                'units': 'kg m-2 s-1',
            }
        ],
        [
            {
                'short_name': 'tas',
                'mip': 'mon',
                'project': 'CORDEX',
                'dataset': 'MOHC-HadGEM3-RA',
                'product': 'output',
                'domain': 'AFR-44',
                'driver': 'ECMWF-ERAINT',
                'exp': 'evaluation',
                'ensemble': 'r1i1p1',
                'institute': 'MOHC',
                'rcm_version': 'v1',
                'timerange': '1991/1993',
            },
            {
                # Added from CMOR table
                'frequency': 'mon',
                'long_name': 'Near-Surface Air Temperature',
                'modeling_realm': ['atmos'],
                'original_short_name': 'tas',
                'standard_name': 'air_temperature',
                'timerange': '1991/1993',
                'units': 'K',
            },
        ],
    ],
)
def test_augment_facets(session, facets, added_facets):
    """Test correct addition of extra facets to an fx dataset."""
    expected_facets = dict(facets)
    expected_facets.update(added_facets)

    dataset = Dataset(**facets)
    dataset.session = session
    dataset._augment_facets()
    assert dataset.facets == expected_facets


def test_from_recipe(session, tmp_path):
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
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tas',
        short_name='tas',
        dataset='dataset1',
        project='CMIP5',
        mip='Amon',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.session = session

    print(Dataset.from_recipe(recipe, session))
    print([dataset])
    assert Dataset.from_recipe(recipe, session) == [dataset]


def test_from_recipe_advanced(session, tmp_path):
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
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            project='CMIP6',
            mip='Amon',
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
            alias='dataset1',
            recipe_dataset_index=0,
        ),
    ]
    for dataset in datasets:
        dataset.session = session

    assert Dataset.from_recipe(recipe, session) == datasets


def test_from_recipe_with_ranges(session, tmp_path):
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
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    datasets = [
        Dataset(
            diagnostic='diagnostic1',
            variable_group='ta',
            short_name='ta',
            dataset='dataset1',
            ensemble='r1i1p1',
            project='CMIP6',
            mip='Amon',
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
            alias='r2i1p1',
            recipe_dataset_index=1,
        ),
    ]
    for dataset in datasets:
        dataset.session = session

    assert Dataset.from_recipe(recipe, session) == datasets


def test_from_recipe_with_supplementary(session, tmp_path):
    recipe_txt = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', ensemble: r1i1p1}

    diagnostics:
      diagnostic1:
        variables:
          tos:
            project: CMIP5
            mip: Omon
            supplementary_variables:
              - short_name: sftof
                mip: fx
    """)
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tos',
        short_name='tos',
        dataset='dataset1',
        ensemble='r1i1p1',
        project='CMIP5',
        mip='Omon',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.supplementaries = [
        Dataset(
            short_name='sftof',
            dataset='dataset1',
            ensemble='r1i1p1',
            project='CMIP5',
            mip='fx',
        ),
    ]
    dataset.session = session

    assert Dataset.from_recipe(recipe, session) == [dataset]


def test_from_recipe_with_skip_supplementary(session, tmp_path):
    recipe_txt = textwrap.dedent("""

    datasets:
      - {dataset: 'dataset1', ensemble: r1i1p1}

    diagnostics:
      diagnostic1:
        variables:
          tos:
            project: CMIP5
            mip: Omon
            supplementary_variables:
              - short_name: sftof
                mip: fx
              - short_name: areacello
                skip: true
    """)
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tos',
        short_name='tos',
        dataset='dataset1',
        ensemble='r1i1p1',
        project='CMIP5',
        mip='Omon',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.supplementaries = [
        Dataset(
            short_name='sftof',
            dataset='dataset1',
            ensemble='r1i1p1',
            project='CMIP5',
            mip='fx',
        ),
    ]
    dataset.session = session

    assert Dataset.from_recipe(recipe, session) == [dataset]


def test_from_recipe_with_automatic_supplementary(session, tmp_path,
                                                  monkeypatch):

    def _find_files(self):
        if self.facets['short_name'] == 'areacello':
            file = esmvalcore.local.LocalFile()
            file.facets = {
                'short_name': 'areacello',
                'mip': 'fx',
                'project': 'CMIP5',
                'dataset': 'dataset1',
                'ensemble': 'r0i0p0',
                'exp': 'piControl',
                'institute': 'X',
                'product': 'output1',
                'version': 'v2',
            }
            files = [file]
        else:
            files = []
        self._files = files

    monkeypatch.setattr(Dataset, '_find_files', _find_files)
    recipe_txt = textwrap.dedent("""

    preprocessors:
      global_mean:
        area_statistics:
          operator: mean

    datasets:
      - {dataset: 'dataset1', ensemble: r1i1p1}

    diagnostics:
      diagnostic1:
        variables:
          tos:
            project: CMIP5
            mip: Omon
            exp: historical
            preprocessor: global_mean
            version: v1
    """)
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    dataset = Dataset(
        diagnostic='diagnostic1',
        variable_group='tos',
        short_name='tos',
        dataset='dataset1',
        ensemble='r1i1p1',
        exp='historical',
        preprocessor='global_mean',
        project='CMIP5',
        version='v1',
        mip='Omon',
        alias='dataset1',
        recipe_dataset_index=0,
    )
    dataset.supplementaries = [
        Dataset(
            short_name='areacello',
            dataset='dataset1',
            institute='X',
            product='output1',
            ensemble='r0i0p0',
            exp='piControl',
            project='CMIP5',
            version='v2',
            mip='fx',
        ),
    ]
    dataset.session = session

    assert Dataset.from_recipe(recipe, session) == [dataset]


@pytest.mark.parametrize('pattern,result', (
    ['a', False],
    ['*', True],
    ['r?i1p1', True],
    ['r[1-3]i1p1*', True],
))
def test_isglob(pattern, result):
    assert esmvalcore.dataset._isglob(pattern) == result


def mock_find_files(*files):
    files_map = defaultdict(list)
    for file in files:
        files_map[file.facets['short_name']].append(file)

    def find_files(self):
        self.files = files_map[self['short_name']]
        for supplementary in self.supplementaries:
            supplementary.files = files_map[supplementary['short_name']]

    return find_files


def test_from_files(session, monkeypatch):
    rootpath = Path('/path/to/data')
    file1 = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'CAS',
        'FGOALS-g3',
        'historical',
        'r3i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20190827',
        'tas_Amon_FGOALS-g3_historical_r3i1p1f1_gn_199001-199912.nc',
    )
    file1.facets = {
        'activity': 'CMIP',
        'institute': 'CAS',
        'dataset': 'FGOALS-g3',
        'exp': 'historical',
        'mip': 'Amon',
        'ensemble': 'r3i1p1f1',
        'short_name': 'tas',
        'grid': 'gn',
        'version': 'v20190827',
    }
    file2 = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'CAS',
        'FGOALS-g3',
        'historical',
        'r3i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20190827',
        'tas_Amon_FGOALS-g3_historical_r3i1p1f1_gn_200001-200912.nc',
    )
    file2.facets = dict(file1.facets)
    file3 = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'NCC',
        'NorESM2-LM',
        'historical',
        'r1i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20190815',
        'tas_Amon_NorESM2-LM_historical_r1i1p1f1_gn_200001-201412.nc',
    )
    file3.facets = {
        'activity': 'CMIP',
        'institute': 'NCC',
        'dataset': 'NorESM2-LM',
        'exp': 'historical',
        'mip': 'Amon',
        'ensemble': 'r1i1p1f1',
        'short_name': 'tas',
        'grid': 'gn',
        'version': 'v20190815',
    }
    find_files = mock_find_files(file1, file2, file3)
    monkeypatch.setattr(Dataset, 'find_files', find_files)

    dataset = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='*',
    )
    dataset.session = session
    datasets = list(dataset.from_files())

    expected = [
        Dataset(short_name='tas',
                mip='Amon',
                project='CMIP6',
                dataset='FGOALS-g3'),
        Dataset(short_name='tas',
                mip='Amon',
                project='CMIP6',
                dataset='NorESM2-LM'),
    ]
    for expected_ds in expected:
        expected_ds.session = session

    assert all(ds.session == session for ds in datasets)
    assert datasets == expected


def test_from_files_with_supplementary(session, monkeypatch):
    rootpath = Path('/path/to/data')
    file = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'CAS',
        'FGOALS-g3',
        'historical',
        'r3i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20190827',
        'tas_Amon_FGOALS-g3_historical_r3i1p1f1_gn_199001-199912.nc',
    )
    file.facets = {
        'activity': 'CMIP',
        'institute': 'CAS',
        'dataset': 'FGOALS-g3',
        'exp': 'historical',
        'mip': 'Amon',
        'ensemble': 'r3i1p1f1',
        'short_name': 'tas',
        'grid': 'gn',
        'version': 'v20190827',
    }
    afile = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'CAS',
        'FGOALS-g3',
        'historical',
        'r1i1p1f1',
        'fx',
        'tas',
        'gn',
        'v20210615',
        'areacella_fx_FGOALS-g3_historical_r1i1p1f1_gn.nc',
    )
    afile.facets = {
        'activity': 'CMIP',
        'institute': 'CAS',
        'dataset': 'FGOALS-g3',
        'exp': 'historical',
        'mip': 'fx',
        'ensemble': 'r1i1p1f1',
        'short_name': 'areacella',
        'grid': 'gn',
        'version': 'v20210615',
    }
    monkeypatch.setattr(Dataset, 'find_files', mock_find_files(file, afile))

    dataset = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='FGOALS-g3',
        ensemble='*',
    )
    dataset.session = session
    dataset.add_supplementary(short_name='areacella', mip='*', ensemble='*')

    expected = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='FGOALS-g3',
        ensemble='r3i1p1f1',
    )
    expected.session = session
    expected.add_supplementary(
        short_name='areacella',
        mip='fx',
        ensemble='r1i1p1f1',
    )

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.supplementaries)
    assert datasets == [expected]


def test_from_files_with_globs(monkeypatch, session):
    """Test `from_files` with wildcards in dataset and supplementary."""
    rootpath = Path('/path/to/data')
    file = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'BCC',
        'BCC-CSM2-MR',
        'historical',
        'r1i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20181126',
        'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc',
    )
    file.facets = {
        'activity': 'CMIP',
        'dataset': 'BCC-CSM2-MR',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gn',
        'institute': 'BCC',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        'version': 'v20181126',
    }
    afile = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'GMMIP',
        'BCC',
        'BCC-CSM2-MR',
        'hist-resIPO',
        'r1i1p1f1',
        'fx',
        'areacella',
        'gn',
        'v20190613',
        'areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc',
    )
    afile.facets = {
        'activity': 'GMMIP',
        'dataset': 'BCC-CSM2-MR',
        'ensemble': 'r1i1p1f1',
        'exp': 'hist-resIPO',
        'grid': 'gn',
        'institute': 'BCC',
        'mip': 'fx',
        'project': 'CMIP6',
        'short_name': 'areacella',
        'version': 'v20190613',
    }
    dataset = Dataset(
        activity='CMIP',
        dataset='*',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        institute='*',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
    )
    dataset.add_supplementary(
        short_name='areacella',
        mip='fx',
        activity='*',
        exp='*',
    )
    dataset.facets['timerange'] = '*'
    dataset.session = session
    print(dataset)

    monkeypatch.setattr(Dataset, 'find_files', mock_find_files(file, afile))

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.supplementaries)

    expected = Dataset(
        activity='CMIP',
        dataset='BCC-CSM2-MR',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        institute='BCC',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
    )
    expected.add_supplementary(
        short_name='areacella',
        mip='fx',
        activity='GMMIP',
        exp='hist-resIPO',
    )
    expected.facets['timerange'] = '185001/201412'
    expected.session = session

    assert datasets == [expected]


def test_from_files_with_globs_and_missing_facets(monkeypatch, session):
    """Test `from_files` with wildcards and files with missing facets.

    Tests a combination of files with complete facets and missing facets.
    """
    rootpath = Path('/path/to/data')
    file1 = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'BCC',
        'BCC-CSM2-MR',
        'historical',
        'r1i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20181126',
        'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc',
    )
    file1.facets = {
        'activity': 'CMIP',
        'dataset': 'BCC-CSM2-MR',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gn',
        'institute': 'BCC',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        'version': 'v20181126',
    }
    file2 = esmvalcore.local.LocalFile(
        rootpath,
        'tas',
        'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc',
    )
    file2.facets = {
        'short_name': 'tas',
    }

    dataset = Dataset(
        activity='CMIP',
        dataset='*',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        institute='*',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='*',
    )

    dataset.session = session
    print(dataset)

    monkeypatch.setattr(Dataset, 'find_files', mock_find_files(file1, file2))

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.supplementaries)

    expected = Dataset(
        activity='CMIP',
        dataset='BCC-CSM2-MR',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        institute='BCC',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='185001/201412',
    )

    expected.session = session

    assert datasets == [expected]


def test_from_files_with_globs_and_automatic_missing(monkeypatch, session):
    """Test `from_files`.

    Test with wildcards and files with missing facets that can be automatically
    added.
    """
    rootpath = Path('/path/to/data')
    file = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'BCC-CSM2-MR',
        'historical',
        'r1i1p1f1',
        'Amon',
        'tas',
        'gn',
        'v20181126',
        'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc',
    )
    file.facets = {
        'dataset': 'BCC-CSM2-MR',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gn',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        'version': 'v20181126',
    }

    dataset = Dataset(
        activity='CMIP',
        dataset='*',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        institute='*',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='*',
    )

    dataset.session = session
    print(dataset)

    monkeypatch.setattr(Dataset, 'find_files', mock_find_files(file))

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.supplementaries)

    expected = Dataset(
        activity='CMIP',
        dataset='BCC-CSM2-MR',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='gn',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='185001/201412',
    )

    expected.session = session

    assert datasets == [expected]


def test_from_files_with_globs_and_only_missing_facets(monkeypatch, session):
    """Test `from_files` with wildcards and only files with missing facets."""
    rootpath = Path('/path/to/data')
    file = esmvalcore.local.LocalFile(
        rootpath,
        'CMIP6',
        'CMIP',
        'BCC',
        'BCC-CSM2-MR',
        'historical',
        'r1i1p1f1',
        'Amon',
        'tas',
        'v20181126',
        'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc',
    )
    file.facets = {
        'activity': 'CMIP',
        'dataset': 'BCC-CSM2-MR',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'institute': 'BCC',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        'version': 'v20181126',
    }

    dataset = Dataset(
        activity='CMIP',
        dataset='*',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='*',
        institute='*',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='*',
    )

    dataset.session = session
    print(dataset)

    monkeypatch.setattr(Dataset, 'find_files', mock_find_files(file))

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.supplementaries)

    expected = Dataset(
        activity='CMIP',
        dataset='BCC-CSM2-MR',
        ensemble='r1i1p1f1',
        exp='historical',
        grid='*',
        institute='BCC',
        mip='Amon',
        project='CMIP6',
        short_name='tas',
        timerange='*',
    )

    expected.session = session

    assert datasets == [expected]


def test_match():
    dataset1 = Dataset(
        short_name='areacella',
        ensemble=['r1i1p1f1'],
        exp='historical',
        modeling_realm=['atmos', 'land'],
    )
    dataset2 = Dataset(
        short_name='tas',
        ensemble='r1i1p1f1',
        exp=['historical', 'ssp585'],
        modeling_realm=['atmos'],
    )

    score = dataset1._match(dataset2)
    assert score == 3


def test_remove_duplicate_supplementaries():
    dataset = Dataset(
        dataset='dataset1',
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        exp='historical',
    )
    supplementary1 = dataset.copy(short_name='areacella')
    supplementary2 = supplementary1.copy()
    supplementary1.facets['exp'] = '1pctCO2'
    dataset.supplementaries = [supplementary1, supplementary2]

    dataset._remove_duplicate_supplementaries()

    assert len(dataset.supplementaries) == 1
    assert dataset.supplementaries[0] == supplementary2


def test_remove_not_found_supplementaries():
    dataset = Dataset(
        dataset='dataset1',
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        exp='historical',
    )
    dataset.add_supplementary(short_name='areacella', mip='fx', exp='*')
    dataset._remove_unexpanded_supplementaries()

    assert len(dataset.supplementaries) == 0


def test_concatenating_historical_and_future_exps(mocker):
    mocker.patch.object(Dataset, 'files', True)
    dataset = Dataset(
        dataset='dataset1',
        short_name='tas',
        mip='Amon',
        frequency='mon',
        project='CMIP6',
        exp=['historical', 'ssp585'],
    )
    dataset.add_supplementary(short_name='areacella', mip='fx', frequency='fx')
    dataset._fix_fx_exp()

    assert len(dataset.supplementaries) == 1
    assert dataset.facets['exp'] == ['historical', 'ssp585']
    assert dataset.supplementaries[0].facets['exp'] == 'historical'


def test_from_recipe_with_glob(tmp_path, session, mocker):
    recipe_txt = textwrap.dedent("""

    diagnostics:
      diagnostic1:
        variables:
          tas:
            project: CMIP5
            mip: Amon
            additional_datasets:
              - {dataset: '*'}
    """)
    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text(recipe_txt, encoding='utf-8')

    session['drs']['CMIP5'] = 'ESGF'

    filenames = [
        "cmip5/output1/CSIRO-QCCCE/CSIRO-Mk3-6-0/rcp85/mon/atmos/Amon/r4i1p1/"
        "v20120323/tas_Amon_CSIRO-Mk3-6-0_rcp85_r4i1p1_200601-210012.nc",
        "cmip5/output1/NIMR-KMA/HadGEM2-AO/historical/mon/atmos/Amon/r1i1p1/"
        "v20130815/tas_Amon_HadGEM2-AO_historical_r1i1p1_186001-200512.nc",
    ]

    mocker.patch.object(
        esmvalcore.local,
        '_get_input_filelist',
        autospec=True,
        spec_set=True,
        return_value=(filenames, []),
    )

    definitions = [
        {
            'diagnostic': 'diagnostic1',
            'variable_group': 'tas',
            'dataset': 'CSIRO-Mk3-6-0',
            'project': 'CMIP5',
            'mip': 'Amon',
            'short_name': 'tas',
            'alias': 'CSIRO-Mk3-6-0',
            'recipe_dataset_index': 0,
        },
        {
            'diagnostic': 'diagnostic1',
            'variable_group': 'tas',
            'dataset': 'HadGEM2-AO',
            'project': 'CMIP5',
            'mip': 'Amon',
            'short_name': 'tas',
            'alias': 'HadGEM2-AO',
            'recipe_dataset_index': 1,
        },
    ]
    expected = []
    for facets in definitions:
        dataset = Dataset(**facets)
        dataset.session = session
        expected.append(dataset)

    datasets = Dataset.from_recipe(recipe, session)
    print("Expected:", expected)
    print("Got:", datasets)
    assert all(ds.session == session for ds in datasets)
    assert datasets == expected


def test_from_ranges():
    dataset = Dataset(ensemble='r(1:2)i1p1f1')
    expected = [
        Dataset(ensemble='r1i1p1f1'),
        Dataset(ensemble='r2i1p1f1'),
    ]
    assert dataset.from_ranges() == expected


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


def test_expand_ensemble_list_ok():
    dataset = Dataset(ensemble=['r0i0p0', 'r1i1p1'])

    expected = [['r0i0p0', 'r1i1p1']]

    assert dataset._expand_range('ensemble') == expected


def test_expand_ensemble_nolist():
    dataset = Dataset(
        dataset='XYZ',
        ensemble=['r1i1p1', 'r(1:2)i1p1'],
    )

    with pytest.raises(RecipeError):
        dataset._expand_range('ensemble')


def create_esgf_file(timerange, version):
    """Prepare some fake ESGF search results."""
    json = {
        'dataset_id':
        'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical'
        f'.r1i1p1f1.Amon.tas.gr.{version}|esgf-data1.llnl.gov',
        'dataset_id_template_': [
            '%(mip_era)s.%(activity_drs)s.%(institution_id)s.'
            '%(source_id)s.%(experiment_id)s.%(member_id)s.'
            '%(table_id)s.%(variable_id)s.%(grid_label)s'
        ],
        'project': ['CMIP6'],
        'size':
        4745571,
        'source_id': ['EC-Earth3'],
        'title':
        f'tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_{timerange}.nc',
        'url': [
            'http://esgf-data1.llnl.gov/thredds/fileServer/css03_data'
            '/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical'
            f'/r1i1p1f1/Amon/tas/gr/{version}/tas_Amon_EC-Earth3'
            f'_historical_r1i1p1f1_gr_{timerange}.nc'
            '|application/netcdf|HTTPServer',
        ],
    }

    results = [pyesgf.search.results.FileResult(json=json, context=None)]

    return ESGFFile(results)


@pytest.fixture
def dataset():
    dataset = Dataset(
        project='CMIP6',
        mip='Amon',
        frequency='mon',
        short_name='tas',
        dataset='EC.-Earth3',
        exp='historical',
        ensemble='r1i1p1f1',
        grid='gr',
        timerange='1850/1851',
        alias='CMIP6_EC-Eeath3_tas',
    )
    dataset.session = {
        'search_esgf': 'when_missing',
        'download_dir': Path('/download_dir'),
        'rootpath': None,
        'drs': {},
    }
    return dataset


@pytest.mark.parametrize("local_availability", ['all', 'partial', 'none'])
def test_find_files(mocker, dataset, local_availability):
    """Test `find_files`."""
    esgf_files = [
        create_esgf_file(version='v1', timerange='185001-185012'),
        create_esgf_file(version='v1', timerange='185101-185112'),
    ]

    local_dir = Path('/local_dir')

    # Local files can cover the entire period, part of it, or nothing
    local_file_options = {
        'all': [f.local_file(local_dir) for f in esgf_files],
        'partial': [esgf_files[1].local_file(local_dir)],
        'none': [],
    }
    local_files = local_file_options[local_availability]

    mocker.patch.object(
        esmvalcore.dataset.Dataset,
        'augment_facets',
        autospec=True,
    )

    mocker.patch.object(
        esmvalcore.dataset.local,
        'find_files',
        autospec=True,
        return_value=(list(local_files), []),
    )
    mocker.patch.object(
        esmvalcore.dataset.esgf,
        'find_files',
        autospec=True,
        return_value=list(esgf_files),
    )

    expected = {
        'all': local_files,
        'partial': local_files + esgf_files[:1],
        'none': esgf_files,
    }
    assert dataset.files == expected[local_availability]


def test_find_files_wildcard_timerange(mocker, dataset):
    """Test that `find_files` works with a '*' in the timerange."""
    dataset.facets['timerange'] = '*'

    esgf_files = [
        create_esgf_file(version='v1', timerange='185001-185012'),
        create_esgf_file(version='v1', timerange='185101-185112'),
    ]
    local_files = []

    mocker.patch.object(
        esmvalcore.dataset.Dataset,
        'augment_facets',
        autospec=True,
    )

    mocker.patch.object(
        esmvalcore.dataset.local,
        'find_files',
        autospec=True,
        return_value=(local_files, []),
    )
    mocker.patch.object(
        esmvalcore.dataset.esgf,
        'find_files',
        autospec=True,
        return_value=list(esgf_files),
    )

    assert dataset.files == esgf_files
    assert dataset.facets['timerange'] == '185001/185112'


def test_find_files_outdated_local(mocker, dataset):
    """Test newer files from ESGF are found when local data is incomplete."""
    esgf_files = [
        create_esgf_file(version='v2', timerange='185001-185012'),
        create_esgf_file(version='v2', timerange='185101-185112'),
    ]

    local_dir = Path('/local_dir')
    local_files = [
        create_esgf_file(version='v1',
                         timerange='185001-185012').local_file(local_dir),
    ]

    mocker.patch.object(
        esmvalcore.dataset.Dataset,
        'augment_facets',
        autospec=True,
    )

    mocker.patch.object(
        esmvalcore.dataset.local,
        'find_files',
        autospec=True,
        return_value=(local_files, []),
    )
    mocker.patch.object(
        esmvalcore.dataset.esgf,
        'find_files',
        autospec=True,
        return_value=list(esgf_files),
    )

    assert dataset.files == esgf_files


@pytest.mark.parametrize(
    'project',
    ['CESM', 'EMAC', 'ICON', 'IPSLCM', 'OBS', 'OBS6', 'ana4mips', 'native6'],
)
def test_find_files_non_esgf_projects(mocker, project, monkeypatch):
    """Test that find_files does never download files for non-ESGF projects."""
    monkeypatch.setitem(CFG, 'search_esgf', 'always')
    mock_local_find_files = mocker.patch.object(
        esmvalcore.dataset.local,
        'find_files',
        autospec=True,
        return_value=(mock.sentinel.files, mock.sentinel.file_globs),
    )
    mock_esgf_find_files = mocker.patch.object(
        esmvalcore.dataset.esgf,
        'find_files',
        autospec=True,
    )

    tas = Dataset(
        short_name='tas',
        mip='Amon',
        project=project,
        dataset='MY_DATASET',
        timerange='2000/2000',
        account='account',
        case='case',
        channel='channel',
        dir='dir',
        exp='amip',
        freq='freq',
        gcomp='gcomp',
        group='group',
        ipsl_varname='ipsl_varname',
        model='model',
        out='out',
        root='root',
        scomp='scomp',
        simulation='simulation',
        status='status',
        string='string',
        tag='tag',
        tdir='tdir',
        tier=3,
        tperiod='tperiod',
        type='sat',
        var_type='var_type',
        version=1,
    )
    tas.augment_facets()
    tas.find_files()

    mock_local_find_files.assert_called_once()
    mock_esgf_find_files.assert_not_called()

    assert tas.files == mock.sentinel.files
    assert tas._file_globs == mock.sentinel.file_globs


def test_set_version():
    dataset = Dataset(short_name='tas')
    dataset.add_supplementary(short_name='areacella')
    file_v1 = esmvalcore.local.LocalFile('/path/to/v1/tas.nc')
    file_v1.facets['version'] = 'v1'
    file_v2 = esmvalcore.local.LocalFile('/path/to/v2/tas.nc')
    file_v2.facets['version'] = 'v2'
    afile = esmvalcore.local.LocalFile('/path/to/v3/areacella.nc')
    afile.facets['version'] = 'v3'
    dataset.files = [file_v2, file_v1]
    dataset.supplementaries[0].files = [afile]
    dataset.set_version()
    assert dataset.facets['version'] == ['v1', 'v2']
    assert dataset.supplementaries[0].facets['version'] == 'v3'


@pytest.mark.parametrize('timerange', ['*', '185001/*', '*/185112'])
def test_update_timerange_from_esgf(mocker, timerange):
    esgf_files = [
        create_esgf_file(version='v20200310', timerange='185001-185012'),
        create_esgf_file(version='v20200310', timerange='185101-185112'),
    ]
    variable = {
        'project': 'CMIP6',
        'mip': 'Amon',
        'frequency': 'mon',
        'short_name': 'tas',
        'dataset': 'EC.-Earth3',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gr',
        'timerange': timerange,
    }
    mocker.patch.object(
        Dataset,
        'files',
        new_callable=mocker.PropertyMock,
        return_value=esgf_files,
    )
    dataset = Dataset(**variable)
    dataset._update_timerange()

    assert dataset['timerange'] == '185001/185112'


TEST_YEAR_FORMAT = [
    ('1/301', '0001/0301'),
    ('10/P2Y', '0010/P2Y'),
    ('P2Y/10', 'P2Y/0010'),
]


@pytest.mark.parametrize('input_time,output_time', TEST_YEAR_FORMAT)
def test_update_timerange_year_format(session, input_time, output_time):
    variable = {
        'project': 'CMIP6',
        'mip': 'Amon',
        'short_name': 'tas',
        'dataset': 'HadGEM3-GC31-LL',
        'exp': 'historical',
        'ensemble': 'r2i1p1f1',
        'grid': 'gr',
        'timerange': input_time
    }
    dataset = Dataset(**variable)
    dataset.session = session
    dataset._update_timerange()
    assert dataset['timerange'] == output_time


@pytest.mark.parametrize('search_esgf', ['never', 'when_missing', 'always'])
def test_update_timerange_no_files(session, search_esgf):
    session['search_esgf'] = search_esgf
    variable = {
        'alias': 'CMIP6',
        'project': 'CMIP6',
        'mip': 'Amon',
        'short_name': 'tas',
        'original_short_name': 'tas',
        'dataset': 'HadGEM3-GC31-LL',
        'exp': 'historical',
        'ensemble': 'r2i1p1f1',
        'grid': 'gr',
        'timerange': '*/2000',
    }
    dataset = Dataset(**variable)
    dataset.files = []
    msg = r"Missing data for Dataset: tas, Amon, CMIP6, HadGEM3-GC31-LL.*"
    with pytest.raises(InputFilesNotFound, match=msg):
        dataset._update_timerange()


def test_update_timerange_typeerror():
    dataset = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='dataset1',
        timerange=42,
    )
    msg = r"timerange should be a string, got '42'"
    with pytest.raises(TypeError, match=msg):
        dataset._update_timerange()


def test_load(mocker, session):
    dataset = Dataset(
        short_name='chl',
        mip='Oyr',
        project='CMIP5',
        dataset='CanESM2',
        exp='historical',
        frequency='yr',
        timerange='2000/2005',
        ensemble='r1i1p1',
    )
    dataset.session = session
    output_file = Path('/path/to/output.nc')
    fix_dir_prefix = Path(
        session.preproc_dir,
        'fixed_files',
        'chl_Oyr_CMIP5_CanESM2_historical_r1i1p1_',
    )
    _get_output_file = mocker.patch.object(
        esmvalcore.dataset,
        '_get_output_file',
        create_autospec=True,
        return_value=output_file,
    )
    args = {}
    order = []

    def mock_preprocess(items, step, input_files, output_file, debug,
                        **kwargs):
        order.append(step)
        args[step] = kwargs
        return items

    mocker.patch.object(esmvalcore.dataset, 'preprocess', mock_preprocess)

    items = [mocker.sentinel.file]
    dataset.files = items

    cube = dataset.load()

    assert cube == items[0]

    load_order = [
        'fix_file',
        'load',
        'fix_metadata',
        'concatenate',
        'cmor_check_metadata',
        'clip_timerange',
        'fix_data',
        'cmor_check_data',
        'add_supplementary_variables',
    ]
    assert order == load_order

    load_args = {
        'load': {
            'ignore_warnings': None,
        },
        'fix_file': {
            'add_unique_suffix': True,
            'session': session,
            'dataset': 'CanESM2',
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'frequency': 'yr',
            'mip': 'Oyr',
            'output_dir': fix_dir_prefix,
            'project': 'CMIP5',
            'short_name': 'chl',
            'timerange': '2000/2005',
        },
        'fix_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'session': session,
            'dataset': 'CanESM2',
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'frequency': 'yr',
            'mip': 'Oyr',
            'project': 'CMIP5',
            'short_name': 'chl',
            'timerange': '2000/2005',
        },
        'cmor_check_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'clip_timerange': {
            'timerange': '2000/2005',
        },
        'fix_data': {
            'check_level': CheckLevels.DEFAULT,
            'session': session,
            'dataset': 'CanESM2',
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'frequency': 'yr',
            'mip': 'Oyr',
            'project': 'CMIP5',
            'short_name': 'chl',
            'timerange': '2000/2005',
        },
        'cmor_check_data': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'concatenate': {
            'check_level': CheckLevels.DEFAULT,
        },
        'add_supplementary_variables': {
            'supplementary_cubes': [],
        },
    }

    assert args == load_args

    _get_output_file.assert_called_with(dataset.facets, session.preproc_dir)


def test_load_fail(session):
    dataset = Dataset()
    dataset.session = session
    dataset.session['search_esgf'] = 'when_missing'
    dataset.files = []
    with pytest.raises(InputFilesNotFound):
        dataset.load()
