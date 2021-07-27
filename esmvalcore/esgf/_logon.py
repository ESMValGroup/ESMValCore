"""Functions for logging on to ESGF."""
import logging
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
        manager.logon(**cfg['logon'])
        logger.info("Logged %s", "on" if manager.is_logged_on() else "off")

    return manager
