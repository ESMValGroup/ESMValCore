import os.path
from collections import defaultdict

import yaml

from esmvalcore._config import _esgf_pyclient

CREDENTIALS = {
    'hostname': 'esgf-data.dkrz.de',
    'username': 'cookiemonster',
    'password': 'Welcome01',
}

DEFAULT_CONFIG: dict = {
    'logon': {
        'interactive': False,
        'bootstrap': True,
    },
    'search_connection': {
        'url': 'http://esgf-node.llnl.gov/esg-search',
        'distrib': True,
        'timeout': 120,
        'cache': os.path.expanduser('~/.pyesgf-cache'),
        'expire_after': 86400
    },
    'preferred_hosts': [],
}
DEFAULT_CONFIG['logon'].update(CREDENTIALS)


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


def test_default_config(monkeypatch, tmp_path):
    """Test that _load_esgf_pyclient_config returns the default config."""
    monkeypatch.setattr(_esgf_pyclient, 'CONFIG_FILE',
                        tmp_path / 'non-existent.yml')

    def get_keyring_credentials():
        return CREDENTIALS

    monkeypatch.setattr(_esgf_pyclient, 'get_keyring_credentials',
                        get_keyring_credentials)

    cfg = _esgf_pyclient._load_esgf_pyclient_config()
    print(cfg)

    assert cfg == DEFAULT_CONFIG
