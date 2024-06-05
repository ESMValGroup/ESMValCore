
from copy import deepcopy
from pathlib import Path

import pytest

import esmvalcore
import esmvalcore.config._config_object
from esmvalcore.config import CFG


@pytest.fixture
def cfg_default(monkeypatch):
    path = Path(esmvalcore.__file__).parent / 'config' / 'config_defaults'
    monkeypatch.setattr(
        esmvalcore.config._config_object,
        'CONFIG_DIRS',
        {'defaults': path},
    )
    cfg = esmvalcore.config._config_object.get_global_config()
    return cfg


@pytest.fixture
def session(tmp_path: Path, cfg_default, monkeypatch):
    for key, value in cfg_default.items():
        monkeypatch.setitem(CFG, key, deepcopy(value))
    monkeypatch.setitem(CFG, 'rootpath', {'default': {tmp_path: 'default'}})
    monkeypatch.setitem(CFG, 'output_dir', tmp_path / 'esmvaltool_output')
    return CFG.start_session('recipe_test')
