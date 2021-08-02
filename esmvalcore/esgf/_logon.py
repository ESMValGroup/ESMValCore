"""Functions for logging on to ESGF."""
import logging
import ssl
from functools import lru_cache

import pyesgf.logon

from esmvalcore._config._esgf_pyclient import _load_esgf_pyclient_config

logger = logging.getLogger(__name__)


@lru_cache(None)
def get_manager():
    """Return a logon manager."""
    return pyesgf.logon.LogonManager()


def logon():
    """Log on to ESGF and return a LogonManager."""
    cfg = _load_esgf_pyclient_config()
    manager = get_manager()

    if not manager.is_logged_on():
        if (cfg['logon'].get('interactive')
                or {'hostname', 'username', 'password'} == set(cfg['logon'])):
            manager.logon(**cfg['logon'])
            if manager.is_logged_on():
                logger.info("Logged on to ESGF")
            else:
                logger.warning("Failed to log on to ESGF, data "
                               "availability will be limited.")

    return manager


def get_credentials():
    """Return ESGF credentials."""
    manager = logon()
    if manager.is_logged_on():
        credentials = manager.esgf_credentials
    else:
        credentials = None
    return credentials


def get_ssl_context():
    """Get an SSL context."""
    credentials = get_credentials()
    if credentials:
        sslcontext = ssl.create_default_context(
            purpose=ssl.Purpose.CLIENT_AUTH)
        sslcontext.load_cert_chain(credentials)
    else:
        sslcontext = None
    return sslcontext
