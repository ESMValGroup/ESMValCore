from pathlib import Path

import pytest
import yaml

from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.cmor.table import __file__ as root
from esmvalcore.cmor.table import read_cmor_tables

CUSTOM_CFG_DEVELOPER = {
    'custom': {'cmor_path': Path(root).parent / 'tables' / 'custom'},
    'CMIP6': {
        'cmor_strict': True,
        'input_dir': {'default': '/'},
        'input_file': '*.nc',
        'output_file': 'out.nc',
        'cmor_type': 'CMIP6',
    },
}


def test_read_cmor_tables():
    """Test that the function `read_cmor_tables` loads the tables correctly."""
    table_path = Path(root).parent / 'tables'

    for project in 'CMIP5', 'CMIP6':
        table = CMOR_TABLES[project]
        assert Path(
            table._cmor_folder) == table_path / project.lower() / 'Tables'
        assert table.strict is True

    project = 'OBS'
    table = CMOR_TABLES[project]
    assert Path(table._cmor_folder) == table_path / 'cmip5' / 'Tables'
    assert table.strict is False

    project = 'OBS6'
    table = CMOR_TABLES[project]
    assert Path(table._cmor_folder) == table_path / 'cmip6' / 'Tables'
    assert table.strict is False

    project = 'obs4MIPs'
    table = CMOR_TABLES[project]
    assert Path(table._cmor_folder) == table_path / 'obs4mips' / 'Tables'
    assert table.strict is False


@pytest.mark.parametrize('behaviour', ['current', 'deprecated'])
def test_read_custom_cmor_tables(tmp_path, behaviour):
    """Test reading of custom CMOR tables."""
    cfg_file = tmp_path / 'config-developer.yml'
    if behaviour == 'deprecated':
        cfg_file = CUSTOM_CFG_DEVELOPER
    else:
        with cfg_file.open('w', encoding='utf-8') as file:
            yaml.safe_dump(CUSTOM_CFG_DEVELOPER, file)

    read_cmor_tables(cfg_file)

    assert len(CMOR_TABLES) == 2
    assert 'CMIP6' in CMOR_TABLES
    assert 'custom' in CMOR_TABLES

    custom_table = CMOR_TABLES['custom']
    assert (Path(custom_table._cmor_folder) ==
            Path(root).parent / 'tables' / 'custom')
    assert (Path(custom_table._coordinates_file) ==
            Path(root).parent / 'tables' / 'custom' / 'CMOR_coordinates.dat')

    cmip6_table = CMOR_TABLES['CMIP6']
    assert cmip6_table.default is custom_table

    # Restore default tables
    read_cmor_tables()
