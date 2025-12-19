from pathlib import Path

import pytest
import yaml

from esmvalcore.config import _esgf_pyclient

DEFAULT_CONFIG: dict = {
    "search_connection": {
        "urls": [
            "https://esgf.ceda.ac.uk/esg-search",
            "https://esgf-data.dkrz.de/esg-search",
            "https://esgf-node.ipsl.upmc.fr/esg-search",
            "https://esg-dn1.nsc.liu.se/esg-search",
            "https://esgf.nci.org.au/esg-search",
            "https://esgf.nccs.nasa.gov/esg-search",
            "https://esgdata.gfdl.noaa.gov/esg-search",
        ],
        "distrib": True,
        "timeout": 120,
        "cache": Path.home()
        / ".esmvaltool"
        / "cache"
        / "pyesgf-search-results",
        "expire_after": 86400,
    },
}


def test_read_config_file(monkeypatch, tmp_path):
    """Test function read_config_file."""
    cfg_file = tmp_path / "esgf-pyclient.yml"
    monkeypatch.setattr(_esgf_pyclient, "CONFIG_FILE", cfg_file)

    reference = {
        "logon": {"interactive": True},
    }
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(reference, file)

    cfg = _esgf_pyclient.read_config_file()
    assert cfg == reference


def test_read_v25_config_file(monkeypatch, tmp_path):
    """Test function read_config_file for v2.5 and earlier.

    For v2.5 and earlier, the ESGF config file contained a single `url`
    instead of a list of `urls` to specify the ESGF index node.
    """
    cfg_file = tmp_path / "esgf-pyclient.yml"
    monkeypatch.setattr(_esgf_pyclient, "CONFIG_FILE", cfg_file)

    cfg_file_content = {
        "search_connection": {"url": "https://some.host/path"},
    }
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg_file_content, file)

    reference = {
        "search_connection": {
            "urls": [
                "https://some.host/path",
            ],
        },
    }

    cfg = _esgf_pyclient.read_config_file()
    assert cfg == reference


@pytest.mark.parametrize("with_keyring_creds", [True, False])
def test_default_config(monkeypatch, mocker, tmp_path, with_keyring_creds):
    """Test that load_esgf_pyclient_config returns the default config."""
    monkeypatch.setattr(
        _esgf_pyclient,
        "CONFIG_FILE",
        tmp_path / "non-existent.yml",
    )

    cfg = _esgf_pyclient.load_esgf_pyclient_config()
    assert cfg == DEFAULT_CONFIG
