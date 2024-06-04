
from pathlib import Path

import pytest

import esmvalcore
import esmvalcore.config._config_object
from esmvalcore.config import Config


@pytest.fixture
def cfg_default(monkeypatch):
    path = Path(esmvalcore.__file__).parent / 'config' / 'config_defaults'
    monkeypatch.setattr(
        esmvalcore.config._config_object,
        'CONFIG_DIRS',
        {'defaults': path},
    )
    cfg = Config._from_global_paths()
    return cfg


@pytest.fixture
def session(tmp_path: Path, cfg_default, monkeypatch):
    monkeypatch.setitem(
        cfg_default, 'rootpath', {'default': {tmp_path: 'default'}}
    )
    session = cfg_default.start_session('recipe_test')
    session['output_dir'] = tmp_path / 'esmvaltool_output'
    return session
