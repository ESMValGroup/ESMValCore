from pathlib import Path

import esmvalcore
from esmvalcore._config import read_config_user_file


def test_read_config_user():
    config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    cfg = read_config_user_file(config_file, 'recipe_test', {'offline': False})
    assert len(cfg) > 1
    assert cfg['offline'] is False
