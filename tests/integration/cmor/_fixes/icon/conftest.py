"""Fixtures for ICON fixes tests."""

import importlib.resources
from pathlib import Path

import pytest
import yaml

import esmvalcore.config
from esmvalcore.cmor._fixes.icon._base_fixes import IconFix


@pytest.fixture(autouse=True)
def tmp_cache_dir(monkeypatch, tmp_path):
    """Use temporary path as cache directory for all tests in this module."""
    monkeypatch.setattr(IconFix, "CACHE_DIR", tmp_path)


@pytest.fixture
def session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session: esmvalcore.config.Session,
) -> esmvalcore.config.Session:
    """Configure ICON data source for all tests in this module."""
    with importlib.resources.as_file(
        importlib.resources.files(esmvalcore.config)
        / "configurations"
        / "data-native-icon.yml",
    ) as config_file:
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    for data_source in cfg["projects"]["ICON"]["data"]:
        cfg["projects"]["ICON"]["data"][data_source]["rootpath"] = tmp_path
    session["projects"]["ICON"]["data"] = cfg["projects"]["ICON"]["data"]
    session["auxiliary_data_dir"] = tmp_path
    return session
