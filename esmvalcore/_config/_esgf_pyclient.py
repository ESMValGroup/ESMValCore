"""esgf-pyclient configuration.

The configuration is read from the file ~/.esmvaltool/esgf-pyclient.yml.

There are four sections in the configuration file:

logon: contains keyword arguments to :func:`pyesgf.logon.LogonManager.logon`
search_connection: contains keyword arguments to
    :class:`pyesgf.search.connection.SearchConnection`
"""
import importlib
import logging
import os
import stat
import textwrap
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

import yaml

from ._config import _normalize_path

keyring: Optional[ModuleType] = None
try:
    keyring = importlib.import_module('keyring')
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / '.esmvaltool' / 'esgf-pyclient.yml'

INSTRUCTIONS = textwrap.dedent("""
ESGF credentials missing, only data that is accessible without
logging in will be available.

See https://esgf.github.io/esgf-user-support/user_guide.html
for instructions on how to create an account if you do not have
one yet.

Next, configure your system so esmvaltool can use your
credentials. This can be done using the keyring package, or
you can just enter them in {cfg_file}.

keyring
=======
First install the keyring package (requires a supported
backend, see https://pypi.org/project/keyring/):
$ pip install keyring

Next, set your username and password by running the commands:
$ keyring set ESGF hostname
$ keyring set ESGF username
$ keyring set ESGF password

To check that you entered your credentials correctly, run:
$ keyring get ESGF hostname
$ keyring get ESGF username
$ keyring get ESGF password

configuration file
==================
You can store the hostname, username, and password or your OpenID
account in a plain text in the file {cfg_file} like this:

logon:
  hostname: "your-hostname"
  username: "your-username"
  password: "your-password"

or your can configure an interactive log in:

logon:
  interactive: true

Note that storing your password in plain text in the configuration
file is less secure. On shared systems, make sure the permissions
of the file are set so only you can read it, i.e.

$ ls -l {cfg_file}

shows permissions -rw-------.

""".format(cfg_file=CONFIG_FILE))


def get_keyring_credentials():
    """Load credentials from keyring."""
    logon = {}
    if keyring is None:
        return logon

    for key in ['hostname', 'username', 'password']:
        try:
            value = keyring.get_password('ESGF', key)
        except keyring.errors.NoKeyringError:
            # No keyring backend is available
            return logon
        if value is not None:
            logon[key] = value

    return logon


def read_config_file():
    """Read the configuration from file."""
    if CONFIG_FILE.exists():
        logger.info("Loading ESGF configuration from %s", CONFIG_FILE)
        mode = os.stat(CONFIG_FILE).st_mode
        if mode & stat.S_IRWXG or mode & stat.S_IRWXO:
            logger.warning("Correcting unsafe permissions on %s", CONFIG_FILE)
            os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
        with CONFIG_FILE.open() as file:
            cfg = yaml.safe_load(file)
    else:
        logger.info(
            "Using default ESGF configuration, configuration "
            "file %s not present.", CONFIG_FILE)
        cfg = {}

    # For backwards compatibility: prior to v2.6 the configuration file
    # contained a single URL instead of a list of URLs.
    if 'search_connection' in cfg:
        if 'url' in cfg['search_connection']:
            url = cfg['search_connection'].pop('url')
            cfg['search_connection']['urls'] = [url]

    return cfg


def load_esgf_pyclient_config():
    """Load the esgf-pyclient configuration."""
    cfg = {
        # Arguments to
        # https://esgf-pyclient.readthedocs.io/en/latest/api.html#pyesgf.logon.LogonManager.logon
        'logon': {
            'interactive': False,
            'bootstrap': True,
        },
        # Arguments to
        # https://esgf-pyclient.readthedocs.io/en/latest/api.html#pyesgf.search.connection.SearchConnection
        'search_connection': {
            # List of available index nodes: https://esgf.llnl.gov/nodes.html
            # Be careful about the url, not all search urls have CMIP3 data?
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
            'distrib': True,
            'timeout': 120,
            'cache': '~/.esmvaltool/cache/pyesgf-search-results',
            'expire_after': 86400,  # cache expires after 1 day
        },
    }

    keyring_cfg = get_keyring_credentials()
    cfg['logon'].update(keyring_cfg)

    file_cfg = read_config_file()
    for section in ['logon', 'search_connection']:
        cfg[section].update(file_cfg.get(section, {}))

    if 'cache' in cfg['search_connection']:
        cache_file = _normalize_path(cfg['search_connection']['cache'])
        cfg['search_connection']['cache'] = cache_file
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    missing_credentials = []
    for key in ['hostname', 'username', 'password']:
        if key not in cfg['logon']:
            missing_credentials.append(key)

    if missing_credentials and not cfg['logon'].get('interactive'):
        logger.warning(INSTRUCTIONS)

    return cfg


@lru_cache()
def get_esgf_config():
    """Get the esgf-pyclient configuration."""
    return load_esgf_pyclient_config()
