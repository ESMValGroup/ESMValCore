"""Test the `esmvalcore.esgf._logon` module."""
import logging

import pyesgf.logon
import pyesgf.search
import pytest

from esmvalcore.esgf import _logon


def test_get_manager():
    manager = _logon.get_manager()
    assert isinstance(manager, pyesgf.logon.LogonManager)


@pytest.mark.parametrize('credentials', [
    {
        'interactive': True
    },
    {
        'hostname': 'esgf-data.dkrz.de',
        'username': 'cookiemonster',
        'password': 'Welcome01',
        'interactive': False,
    },
])
def test_logon(mocker, caplog, credentials):

    cfg = {'logon': credentials}
    mocker.patch.object(_logon,
                        'get_esgf_config',
                        autospec=True,
                        return_value=cfg)
    manager = mocker.create_autospec(pyesgf.logon.LogonManager,
                                     spec_set=True,
                                     instance=True)
    manager.is_logged_on.side_effect = False, True
    mocker.patch.object(_logon, 'get_manager', return_value=manager)

    caplog.set_level(logging.INFO)

    _logon.logon()

    manager.logon.assert_called_with(**cfg['logon'])
    assert "Logged on to ESGF" in caplog.text


def test_logon_fail_message(mocker, caplog):
    cfg = {'logon': {'interactive': True}}
    mocker.patch.object(_logon,
                        'get_esgf_config',
                        autospec=True,
                        return_value=cfg)
    manager = mocker.create_autospec(pyesgf.logon.LogonManager,
                                     spec_set=True,
                                     instance=True)
    manager.is_logged_on.return_value = False
    mocker.patch.object(_logon, 'get_manager', return_value=manager)

    _logon.logon()

    manager.logon.assert_called_with(**cfg['logon'])
    assert "Failed to log on to ESGF" in caplog.text


@pytest.mark.parametrize('logged_on', [True, False])
def test_get_credentials(mocker, logged_on):

    manager = mocker.create_autospec(pyesgf.logon.LogonManager,
                                     spec_set=False,
                                     instance=True)
    manager.is_logged_on.return_value = logged_on
    manager.esgf_credentials = '/path/to/creds.pem'
    mocker.patch.object(_logon, 'logon', return_value=manager)

    creds = _logon.get_credentials()

    if logged_on:
        assert creds == '/path/to/creds.pem'
    else:
        assert creds is None
