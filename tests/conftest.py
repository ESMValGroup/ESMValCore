from copy import deepcopy
from pathlib import Path

import pytest

import esmvalcore.config._dask
from esmvalcore.config import CFG, Config
from esmvalcore.config._config_object import _get_global_config


@pytest.fixture
def cfg_default():
    """Create a configuration object with default values."""
    cfg = _get_global_config()
    cfg.load_from_dirs([])
    return cfg


@pytest.fixture()
def ignore_existing_user_config(monkeypatch, cfg_default):
    """Ignore user's configuration when running tests."""
    for key, value in cfg_default.items():
        monkeypatch.setitem(CFG, key, deepcopy(value))


@pytest.fixture
def session(tmp_path: Path, cfg_default, monkeypatch):
    """Session object with default settings."""
    for key, value in cfg_default.items():
        monkeypatch.setitem(CFG, key, deepcopy(value))
    monkeypatch.setitem(CFG, "rootpath", {"default": {tmp_path: "default"}})
    monkeypatch.setitem(CFG, "output_dir", tmp_path / "esmvaltool_output")
    return CFG.start_session("recipe_test")


# TODO: remove in v2.14.0
@pytest.fixture(autouse=True)
def ignore_old_config_user(tmp_path, monkeypatch):
    """Ignore potentially existing old config-user.yml file in all tests."""
    nonexistent_config_dir = tmp_path / "nonexistent_config_dir"
    monkeypatch.setattr(
        Config, "_DEFAULT_USER_CONFIG_DIR", nonexistent_config_dir
    )


# TODO: remove in v2.14.0
@pytest.fixture(autouse=True)
def ignore_old_dask_config_file(tmp_path, monkeypatch):
    """Ignore potentially existing old dask.yml file in all tests."""
    nonexistent_file = tmp_path / "nonexistent_file.yml"
    monkeypatch.setattr(
        esmvalcore.config._dask, "CONFIG_FILE", nonexistent_file
    )
