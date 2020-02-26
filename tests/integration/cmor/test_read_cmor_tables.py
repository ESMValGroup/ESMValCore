from pathlib import Path

from esmvalcore._config import read_config_developer_file
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.cmor.table import __file__ as root
from esmvalcore.cmor.table import read_cmor_tables


def test_read_cmor_tables():
    """Test that the funcion `read_cmor_tables` loads the tables correctly."""
    # Read the tables
    read_cmor_tables(read_config_developer_file())

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

    project = 'obs4mips'
    table = CMOR_TABLES[project]
    assert Path(table._cmor_folder) == table_path / 'obs4mips' / 'Tables'
    assert table.strict is False
