import textwrap
from pathlib import Path

import pyesgf
import pytest

import esmvalcore.dataset
import esmvalcore.local
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.config import CFG
from esmvalcore.config._config_object import CFG_DEFAULT
from esmvalcore.dataset import Dataset
from esmvalcore.esgf import ESGFFile
from esmvalcore.exceptions import InputFilesNotFound, RecipeError


@pytest.fixture
def session(tmp_path):
    CFG.clear()
    CFG.update(CFG_DEFAULT)
    CFG['output_dir'] = tmp_path / 'esmvaltool_output'
    return CFG.start_session('recipe_test')


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


def test_repr_ancillary():
    ds = Dataset(dataset='dataset1', short_name='tas')
    ds.add_ancillary(short_name='areacella')

    assert repr(ds) == textwrap.dedent("""
        Dataset:
        {'dataset': 'dataset1', 'short_name': 'tas'}
        ancillaries:
          {'dataset': 'dataset1', 'short_name': 'areacella'}
        """).strip()


def test_short_summary():
    ds = Dataset(
        project='CMIP6',
        dataset='dataset1',
        short_name='tos',
        mip='Omon',
    )
    ds.add_ancillary(short_name='areacello', mip='Ofx')
    ds.add_ancillary(short_name='volcello')
    expected = ("Dataset: tos, Omon, CMIP6, dataset1, "
                "ancillaries: areacello, Ofx; volcello")
    assert ds.summary(shorten=True) == expected


def test_long_summary():
    ds = Dataset(dataset='dataset1', short_name='tas')
    assert ds.summary(shorten=False) == repr(ds)


def test_session_setter():
    ds = Dataset(short_name='tas')
    ds.add_ancillary(short_name='areacella')
    assert ds._session is None
    assert ds.ancillaries[0]._session is None

    ds.session

    assert isinstance(ds.session, esmvalcore.config.Session)
    assert ds.session == ds.ancillaries[0].session


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
                'dataset': 'CanESM2',
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
                'institute': ['CCCma'],
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


@pytest.mark.parametrize('pattern,result', (
    ['a', False],
    ['*', True],
    ['r?i1p1', True],
    ['r[1-3]i1p1*', True],
))
def test_isglob(pattern, result):
    assert esmvalcore.dataset._isglob(pattern) == result


def test_from_files(session, mocker):
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
    dataset = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='*',
    )
    dataset.session = session
    mocker.patch.object(
        Dataset,
        'files',
        new_callable=mocker.PropertyMock,
        side_effect=[[file1, file2, file3]],
    )
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


def test_from_files_with_ancillary(session, mocker):
    rootpath = Path('/path/to/data')
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
    dataset = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='FGOALS-g3',
        ensemble='r3i1p1f1',
    )
    dataset.session = session
    dataset.add_ancillary(short_name='areacella', mip='fx', ensemble='*')
    mocker.patch.object(
        Dataset,
        'files',
        new_callable=mocker.PropertyMock,
        side_effect=[[afile]],
    )

    expected = Dataset(
        short_name='tas',
        mip='Amon',
        project='CMIP6',
        dataset='FGOALS-g3',
        ensemble='r3i1p1f1',
    )
    expected.session = session
    expected.add_ancillary(
        short_name='areacella',
        mip='fx',
        ensemble='r1i1p1f1',
    )

    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.ancillaries)
    assert datasets == [expected]


def test_from_files_with_globs(mocker, session):
    """Test `from_files` with wildcards in dataset and ancillary."""
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
    dataset.add_ancillary(
        short_name='areacella',
        mip='fx',
        activity='*',
        exp='*',
    )
    dataset.facets['timerange'] = '*'
    dataset.session = session
    print(dataset)

    mocker.patch.object(
        Dataset,
        'files',
        new_callable=mocker.PropertyMock,
        # One call to `files` in `_get_available_facets` for 'tas'
        # Two calls to `files` in `_update_timerange` for 'tas'
        # One call to `files` in `_get_available facets` for 'areacella'
        side_effect=[[file], [file], [file], [afile]],
    )
    datasets = list(dataset.from_files())

    assert all(ds.session == session for ds in datasets)
    assert all(ads.session == session for ds in datasets
               for ads in ds.ancillaries)

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
    expected.add_ancillary(
        short_name='areacella',
        mip='fx',
        activity='GMMIP',
        exp='hist-resIPO',
    )
    expected.facets['timerange'] = '185001/201412'
    expected.session = session

    assert datasets == [expected]


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
        'always_search_esgf': False,
        'download_dir': Path('/download_dir'),
        'offline': False,
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


def test_set_version():
    dataset = Dataset(short_name='tas')
    dataset.add_ancillary(short_name='areacella')
    file_v1 = esmvalcore.local.LocalFile('/path/to/v1/tas.nc')
    file_v1.facets['version'] = 'v1'
    file_v2 = esmvalcore.local.LocalFile('/path/to/v2/tas.nc')
    file_v2.facets['version'] = 'v2'
    afile = esmvalcore.local.LocalFile('/path/to/v3/areacella.nc')
    afile.facets['version'] = 'v3'
    dataset.files = [file_v2, file_v1]
    dataset.ancillaries[0].files = [afile]
    dataset.set_version()
    assert dataset.facets['version'] == ['v1', 'v2']
    assert dataset.ancillaries[0].facets['version'] == 'v3'


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


@pytest.mark.parametrize('offline', [True, False])
def test_update_timerange_no_files(session, offline):
    session['offline'] = offline
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
    msg = r"Missing data for CMIP6: tas.*"
    with pytest.raises(InputFilesNotFound, match=msg):
        dataset._update_timerange()


def test_update_timerange_typeerror():
    dataset = Dataset(timerange=42)
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
    fix_dir = Path('/path/to/output_fixed')
    _get_output_file = mocker.patch.object(
        esmvalcore.dataset,
        '_get_output_file',
        create_autospec=True,
        return_value=output_file,
    )
    args = {}
    order = []

    def mock_preprocess(items, step, input_files, **kwargs):
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
        'add_ancillary_variables',
    ]
    assert order == load_order

    load_args = {
        'load': {},
        'fix_file': {
            'dataset': 'CanESM2',
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'frequency': 'yr',
            'mip': 'Oyr',
            'output_dir': fix_dir,
            'project': 'CMIP5',
            'short_name': 'chl',
            'timerange': '2000/2005',
        },
        'fix_metadata': {
            'check_level': CheckLevels.DEFAULT,
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
        'concatenate': {},
        'add_ancillary_variables': {
            'ancillary_cubes': [],
        },
    }

    assert args == load_args

    _get_output_file.assert_called_with(dataset.facets, session.preproc_dir)


def test_load_fail(session):
    dataset = Dataset()
    dataset.session = session
    dataset.session['offline'] = False
    dataset.files = []
    with pytest.raises(InputFilesNotFound):
        dataset.load()
