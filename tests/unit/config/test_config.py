from pathlib import Path

import pytest
import yaml

from esmvalcore._config import _config
from esmvalcore._config._config import (
    _deep_update,
    _load_extra_facets,
    get_extra_facets,
    importlib_files,
)

TEST_DEEP_UPDATE = [
    ([{}], {}),
    ([dict(a=1, b=2), dict(a=3)], dict(a=3, b=2)),
    ([
        dict(a=dict(b=1, c=dict(d=2)), e=dict(f=4, g=5)),
        dict(a=dict(b=2, c=3)),
    ], dict(a=dict(b=2, c=3), e=dict(f=4, g=5))),
]


@pytest.mark.parametrize('dictionaries, expected_merged', TEST_DEEP_UPDATE)
def test_deep_update(dictionaries, expected_merged):
    merged = dictionaries[0]
    for update in dictionaries[1:]:
        merged = _deep_update(merged, update)
    assert expected_merged == merged


BASE_PATH = importlib_files('tests')
BASE_PATH /= Path('sample_data') / Path('extra_facets')  # type: ignore

TEST_LOAD_EXTRA_FACETS = [
    ('test-nonexistent', tuple(), {}),
    ('test-nonexistent', (BASE_PATH / 'simple', ), {}),  # type: ignore
    (
        'test6',
        (BASE_PATH / 'simple', ),  # type: ignore
        dict(PROJECT1=dict(Amon=dict(
            tas=dict(cds_var_name='2m_temperature', source_var_name='2t'),
            psl=dict(cds_var_name='mean_sea_level_pressure',
                     source_var_name='msl'))))),
    (
        'test6',
        (BASE_PATH / 'simple', BASE_PATH / 'override'),  # type: ignore
        dict(PROJECT1=dict(Amon=dict(
            tas=dict(cds_var_name='temperature_2m', source_var_name='t2m'),
            psl=dict(cds_var_name='mean_sea_level_pressure',
                     source_var_name='msl'),
            uas=dict(cds_var_name='10m_u-component_of_neutral_wind',
                     source_var_name='u10n'),
            vas=dict(cds_var_name='v-component_of_neutral_wind_at_10m',
                     source_var_name='10v'),
        )))),
]


@pytest.mark.parametrize('project, extra_facets_dir, expected',
                         TEST_LOAD_EXTRA_FACETS)
def test_load_extra_facets(project, extra_facets_dir, expected):
    extra_facets = _load_extra_facets(project, extra_facets_dir)
    assert extra_facets == expected


def test_get_extra_facets_cmip3():

    variable = {
        'project': 'CMIP3',
        'mip': 'A1',
        'short_name': 'tas',
        'dataset': 'CM3',
    }
    extra_facets = get_extra_facets(**variable, extra_facets_dir=tuple())

    assert extra_facets == {'institute': ['CNRM', 'INM']}


def test_get_extra_facets_cmip5():

    variable = {
        'project': 'CMIP5',
        'mip': 'Amon',
        'short_name': 'tas',
        'dataset': 'ACCESS1-0',
    }
    extra_facets = get_extra_facets(**variable, extra_facets_dir=tuple())

    assert extra_facets == {'institute': ['CSIRO-BOM'], 'product': 'output1'}


def test_load_default_config(monkeypatch):
    """Test that the default configuration can be loaded."""
    project_cfg = {}
    monkeypatch.setattr(_config, 'CFG', project_cfg)
    default_cfg_file = importlib_files('esmvalcore') / 'config-user.yml'
    cfg = _config.read_config_user_file(default_cfg_file, 'recipe_example')

    default_cfg = {
        'download_dir': str(Path.home() / 'climate_data'),
        'auxiliary_data_dir': str(Path.home() / 'auxiliary_data'),
        'max_parallel_tasks': None,
        'offline': True,
        'log_level': 'info',
        'exit_on_warning': False,
        'output_file_type': 'png',
        'remove_preproc_dir': False,
        'compress_netcdf': False,
        'save_intermediary_cubes': False,
        'config_developer_file': None,
        'profile_diagnostic': False,
        'rootpath': {
            'default': [str(Path.home() / 'climate_data')]
        },
        'drs': {
            'CMIP3': 'ESGF',
            'CMIP5': 'ESGF',
            'CMIP6': 'ESGF',
            'CORDEX': 'ESGF',
            'obs4MIPs': 'ESGF'
        },
        'extra_facets_dir': tuple(),
        'run_diagnostic': True,
        'config_file': str(default_cfg_file),
    }
    default_keys = set(
        list(default_cfg) + [
            'output_dir',
            'preproc_dir',
            'work_dir',
            'plot_dir',
            'run_dir',
        ])

    # Check that only allowed keys are in it
    assert default_keys == set(cfg)

    # Check default values
    for key in default_cfg:
        assert cfg[key] == default_cfg[key]

    # Check output directories
    assert cfg['output_dir'].startswith(
        str(Path.cwd() / 'esmvaltool_output' / 'recipe_example'))
    for path in ('preproc', 'work', 'run'):
        assert cfg[path + '_dir'] == str(Path(cfg['output_dir'], path))
    assert cfg['plot_dir'] == str(Path(cfg['output_dir'], 'plots'))

    # Check that projects were configured
    assert project_cfg


def test_rootpath_obs4mips_case_correction(tmp_path, monkeypatch, mocker):
    """Test that the name of the obs4MIPs project is correct in rootpath."""
    monkeypatch.setattr(_config, 'CFG', {})
    mocker.patch.object(_config, 'read_cmor_tables', autospec=True)
    cfg_file = tmp_path / 'config-user.yml'
    cfg_user = {
        'rootpath': {
            'obs4mips': '/path/to/data',
        },
    }
    with cfg_file.open('w') as file:
        yaml.safe_dump(cfg_user, file)

    cfg = _config.read_config_user_file(cfg_file, 'recipe_example')

    assert 'obs4mips' not in cfg['rootpath']
    assert cfg['rootpath']['obs4MIPs'] == ['/path/to/data']


def test_project_obs4mips_case_correction(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(_config, 'CFG', {})
    mocker.patch.object(_config, 'read_cmor_tables', autospec=True)
    cfg_file = tmp_path / 'config-developer.yml'
    cfg_dev = {
        'obs4mips': {},
    }
    with cfg_file.open('w') as file:
        yaml.safe_dump(cfg_dev, file)

    cfg = _config.read_config_developer_file(cfg_file)

    assert 'obs4mips' not in cfg
    assert cfg['obs4MIPs'] == {}
