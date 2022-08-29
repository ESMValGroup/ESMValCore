import textwrap
from pathlib import Path

import pyesgf
import pytest
import yaml

import esmvalcore.dataset
from esmvalcore._config import CFG
from esmvalcore._config._config_object import CFG_DEFAULT
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.dataset import (
    Dataset,
    _path2facets,
    datasets_from_recipe,
    datasets_to_recipe,
)
from esmvalcore.esgf import ESGFFile
from esmvalcore.exceptions import InputFilesNotFound, RecipeError


@pytest.fixture
def session(tmp_path):
    CFG.clear()
    CFG.update(CFG_DEFAULT)
    CFG['output_dir'] = tmp_path
    return CFG.start_session('recipe_test')


def test_repr():

    ds = Dataset(short_name='tas', dataset='dataset1')

    assert repr(ds) == textwrap.dedent("""
        Dataset:
        {'dataset': 'dataset1', 'short_name': 'tas'}
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
                'start_year': 2000,
                'end_year': 2001,
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
                # Updated time range
                'timerange': '2000/2001',
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
    for key in ('start_year', 'end_year'):
        expected_facets.pop(key, None)

    dataset = Dataset(**facets)
    dataset._augment_facets(session)
    assert dataset.facets == expected_facets


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
    for dataset in datasets:
        dataset.session = session

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


def create_esgf_search_results():
    """Prepare some fake ESGF search results."""
    file0 = ESGFFile([
        pyesgf.search.results.FileResult(
            json={
                'dataset_id':
                'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical.r1i1p1f1'
                '.Amon.tas.gr.v20200310|esgf-data1.llnl.gov',
                'dataset_id_template_': [
                    '%(mip_era)s.%(activity_drs)s.%(institution_id)s.' +
                    '%(source_id)s.%(experiment_id)s.%(member_id)s.' +
                    '%(table_id)s.%(variable_id)s.%(grid_label)s'
                ],
                'project': ['CMIP6'],
                'size':
                4745571,
                'source_id': ['EC-Earth3'],
                'title':
                'tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_185001-185012.nc',
                'url': [
                    'http://esgf-data1.llnl.gov/thredds/fileServer/css03_data'
                    '/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical'
                    '/r1i1p1f1/Amon/tas/gr/v20200310/tas_Amon_EC-Earth3'
                    '_historical_r1i1p1f1_gr_185001-185012.nc'
                    '|application/netcdf|HTTPServer',
                ],
            },
            context=None,
        )
    ])
    file1 = ESGFFile([
        pyesgf.search.results.FileResult(
            {
                'dataset_id':
                'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical.r1i1p1f1'
                '.Amon.tas.gr.v20200310|esgf-data1.llnl.gov',
                'dataset_id_template_': [
                    '%(mip_era)s.%(activity_drs)s.%(institution_id)s.' +
                    '%(source_id)s.%(experiment_id)s.%(member_id)s.' +
                    '%(table_id)s.%(variable_id)s.%(grid_label)s'
                ],
                'project': ['CMIP6'],
                'size':
                4740192,
                'source_id': ['EC-Earth3'],
                'title':
                'tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_185101-185112.nc',
                'url': [
                    'http://esgf-data1.llnl.gov/thredds/fileServer/css03_data'
                    '/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical'
                    '/r1i1p1f1/Amon/tas/gr/v20200310/tas_Amon_EC-Earth3'
                    '_historical_r1i1p1f1_gr_185101-185112.nc'
                    '|application/netcdf|HTTPServer',
                ],
            },
            context=None,
        )
    ])

    return [file0, file1]


@pytest.mark.parametrize("local_availability", ['all', 'partial', 'none'])
def test_find_files(mocker, local_availability):

    esgf_files = create_esgf_search_results()

    local_dir = Path('/local_dir')

    # Local files can cover the entire period, part of it, or nothing
    local_file_options = {
        'all': [f.local_file(local_dir) for f in esgf_files],
        'partial': [esgf_files[1].local_file(local_dir)],
        'none': [],
    }
    local_files = local_file_options[local_availability]

    mocker.patch.object(esmvalcore.dataset,
                        'get_input_filelist',
                        autospec=True,
                        return_value=(list(local_files), [], []))
    mocker.patch.object(
        esmvalcore.dataset.esgf,
        'find_files',
        autospec=True,
        return_value=esgf_files,
    )

    variable = {
        'project': 'CMIP6',
        'mip': 'Amon',
        'frequency': 'mon',
        'short_name': 'tas',
        'dataset': 'EC.-Earth3',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gr',
        'timerange': '1850/1851',
        'alias': 'CMIP6_EC-Eeath3_tas',
    }
    dataset = Dataset(**variable)
    session = {
        'offline': False,
        'download_dir': Path('/download_dir'),
        'rootpath': None,
        'drs': {},
        'download_latest_datasets': True,
    }
    dataset._find_files(session)
    input_files = dataset.files

    expected = {
        'all': local_files,
        'partial': local_files + esgf_files[:1],
        'none': esgf_files,
    }
    assert input_files == expected[local_availability]


@pytest.mark.parametrize('timerange', ['*', '185001/*', '*/185112'])
def test_update_timerange_from_esgf(mocker, timerange):

    esgf_files = create_esgf_search_results()
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
    mocker.patch.object(Dataset, 'find_files', create_autospec=True)
    dataset = Dataset(**variable)
    dataset.files = esgf_files
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
    dataset._update_timerange(session)
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
    msg = r"Missing data for: CMIP6, Amon, tas, HadGEM3-GC31-LL.*"
    with pytest.raises(InputFilesNotFound, match=msg):
        dataset._update_timerange(session)


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
    get_output_file = mocker.patch.object(esmvalcore.dataset,
                                          'get_output_file',
                                          create_autospec=True,
                                          return_value=output_file)
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
        'add_fx_variables',
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
        'add_fx_variables': {
            'fx_variables': [],
        },
    }

    assert args == load_args

    get_output_file.assert_called_with(dataset.facets, session.preproc_dir)


def test_path2facets():
    """Test `_path2facets1."""
    filepath = Path("/climate_data/value1/value2/filename.nc")
    drs = "{facet1}/{facet2.lower}"

    expected = {
        'facet1': 'value1',
        'facet2': 'value2',
    }

    result = _path2facets(filepath, drs)

    assert result == expected
