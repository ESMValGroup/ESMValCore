
from copy import deepcopy
from pathlib import Path

import pytest

from esmvalcore.config import CFG


@pytest.fixture
def cfg_default(mocker):
    """Configuration object with defaults."""
    cfg = deepcopy(CFG)
    cfg.restore_default()
    return cfg


@pytest.fixture
def session(tmp_path: Path, cfg_default, monkeypatch):
    """Session object with default settings."""
    for key, value in cfg_default.items():
        monkeypatch.setitem(CFG, key, deepcopy(value))
    monkeypatch.setitem(CFG, 'rootpath', {'default': {tmp_path: 'default'}})
    monkeypatch.setitem(CFG, 'output_dir', tmp_path / 'esmvaltool_output')
    return CFG.start_session('recipe_test')
