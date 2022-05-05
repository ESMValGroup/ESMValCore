"""Functions for logging on to ESGF."""
import logging
from functools import lru_cache

import pyesgf.logon
import pyesgf.search

from .._config._esgf_pyclient import get_esgf_config

logger = logging.getLogger(__name__)


@lru_cache(None)
def get_manager():
    """Return a logon manager."""
    return pyesgf.logon.LogonManager()


def logon():
    """Log on to ESGF and return a LogonManager."""
    cfg = get_esgf_config()
    manager = get_manager()

    if not manager.is_logged_on():
        keys = ['interactive', 'hostname', 'username', 'password']
        if any(cfg['logon'].get(key) for key in keys):
            # only try logging on if it is configured
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
