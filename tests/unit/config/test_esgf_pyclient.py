import copy
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from esmvalcore._config import _esgf_pyclient

DEFAULT_CONFIG: dict = {
    'logon': {
        'interactive': False,
        'bootstrap': True,
    },
    'search_connection': {
        'urls': [
            'https://esgf.ceda.ac.uk/esg-search',
            'https://esgf-node.llnl.gov/esg-search',
            'https://esgf-data.dkrz.de/esg-search',
            'https://esgf-node.ipsl.upmc.fr/esg-search',
            'https://esg-dn1.nsc.liu.se/esg-search',
            'https://esgf.nci.org.au/esg-search',
            'https://esgf.nccs.nasa.gov/esg-search',
            'https://esgdata.gfdl.noaa.gov/esg-search',
        ],
        'distrib':
        True,
        'timeout':
        120,
        'cache':
        str(Path.home() / '.esmvaltool' / 'cache' / 'pyesgf-search-results'),
        'expire_after':
        86400,
    },
}

CREDENTIALS = {
    'hostname': 'esgf-data.dkrz.de',
    'username': 'cookiemonster',
    'password': 'Welcome01',
}


class MockKeyring:
    """Mock keyring module."""

    def __init__(self):
        self.items = defaultdict(dict)

    def set_password(self, service_name, username, password):
        self.items[service_name][username] = password

    def get_password(self, service_name, username):
        return self.items[service_name][username]


def test_get_keyring_credentials(monkeypatch):
    """Test function get_keyring_credentials."""
    keyring = MockKeyring()
    for key, value in CREDENTIALS.items():
        keyring.set_password("ESGF", key, value)
    monkeypatch.setattr(_esgf_pyclient, 'keyring', keyring)

    credentials = _esgf_pyclient.get_keyring_credentials()

    assert credentials == CREDENTIALS


def test_get_keyring_credentials_no_keyring(mocker):

    mocker.patch.object(_esgf_pyclient, 'keyring', None)
    credentials = _esgf_pyclient.get_keyring_credentials()
    assert credentials == {}


def test_get_keyring_credentials_no_backend(mocker):

    keyring = mocker.patch.object(_esgf_pyclient, 'keyring')
    keyring.errors.NoKeyringError = Exception
    keyring.get_password.side_effect = keyring.errors.NoKeyringError
    credentials = _esgf_pyclient.get_keyring_credentials()
    assert credentials == {}


def test_read_config_file(monkeypatch, tmp_path):
    """Test function read_config_file."""
    cfg_file = tmp_path / 'esgf-pyclient.yml'
    monkeypatch.setattr(_esgf_pyclient, 'CONFIG_FILE', cfg_file)

    reference = {
        'logon': {
            'interactive': True
        },
    }
    with cfg_file.open('w') as file:
        yaml.safe_dump(reference, file)

    cfg = _esgf_pyclient.read_config_file()
    assert cfg == reference


def test_read_v25_config_file(monkeypatch, tmp_path):
    """Test function read_config_file for v2.5 and earlier.

    For v2.5 and earlier, the config-file contained a single `url`
    instead of a list of `urls` to specify the ESGF index node.
    """
    cfg_file = tmp_path / 'esgf-pyclient.yml'
    monkeypatch.setattr(_esgf_pyclient, 'CONFIG_FILE', cfg_file)

    cfg_file_content = {
        'search_connection': {
            'url': 'https://some.host/path'
        },
    }
    with cfg_file.open('w') as file:
        yaml.safe_dump(cfg_file_content, file)

    reference = {
        'search_connection': {
            'urls': [
                'https://some.host/path',
            ]
        }
    }

    cfg = _esgf_pyclient.read_config_file()
    assert cfg == reference


@pytest.mark.parametrize('with_keyring_creds', [True, False])
def test_default_config(monkeypatch, mocker, tmp_path, with_keyring_creds):
    """Test that load_esgf_pyclient_config returns the default config."""
    monkeypatch.setattr(_esgf_pyclient, 'CONFIG_FILE',
                        tmp_path / 'non-existent.yml')

    credentials = CREDENTIALS if with_keyring_creds else {}
    mocker.patch.object(
        _esgf_pyclient,
        'get_keyring_credentials',
        autospec=True,
        return_value=credentials,
    )

    cfg = _esgf_pyclient.load_esgf_pyclient_config()

    expected = copy.deepcopy(DEFAULT_CONFIG)
    expected['logon'].update(credentials)

    assert cfg == expected
