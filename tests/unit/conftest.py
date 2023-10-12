import copy

import pytest

from esmvalcore.config import CFG
from esmvalcore.config._config_object import CFG_DEFAULT


@pytest.fixture
def session(tmp_path, monkeypatch):
    for key, value in CFG_DEFAULT.items():
        monkeypatch.setitem(CFG, key, copy.deepcopy(value))
    monkeypatch.setitem(CFG, 'output_dir', tmp_path / 'esmvaltool_output')
    return CFG.start_session('recipe_test')
