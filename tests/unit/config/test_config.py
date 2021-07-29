from pathlib import Path

import pytest

from esmvalcore._config._config import (
    _deep_update,
    _load_extra_facets,
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
