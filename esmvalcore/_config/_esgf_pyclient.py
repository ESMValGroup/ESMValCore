"""esgf-pyclient configuration.

The configuration is read from the file ~/.esmvaltool/esgf-pyclient.yml.

There are four sections in the configuration file:

logon: contains keyword arguments to :func:`pyesgf.logon.LogonManager.logon`
search_connection: contains keyword arguments to :class:`pyesgf.search.connection.SearchConnection`
preferred_hosts: the first host in this list that has a certain file will be used
ignore_hosts: these hosts will not be used
"""
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(None)
def _load_esgf_pyclient_config():

    cfg = {
        # Arguments to
        # https://esgf-pyclient.readthedocs.io/en/latest/api.html#pyesgf.logon.LogonManager.logon
        'logon': {
            'hostname': 'esgf-data.dkrz.de',
            'interactive': False,
            'bootstrap': True,
        },
        # Arguments to
        # https://esgf-pyclient.readthedocs.io/en/latest/api.html#pyesgf.search.connection.SearchConnection
        'search_connection': {
            'url': 'https://esgf-data.dkrz.de/esg-search',
            'distrib': True,
            'timeout': 120,
            'cache': str(Path().home() / '.pyesgf-cache'),
            'expire_after': 86400,  # cache expires after 1 day
        },
    }

    cfg_file = Path().home() / '.esmvaltool' / 'esgf-pyclient.yml'
    with cfg_file.open() as file:
        user_cfg = yaml.safe_load(file)

    if 'logon' not in user_cfg:
        raise ValueError(f"Section 'logon:' missing from {cfg_file}")
    for key in ('username', 'password'):
        if key not in user_cfg['logon'] or not user_cfg['logon'].get(key):
            raise ValueError(
                f"'{key}' missing from section 'logon:' in {cfg_file}")

    for section in ['logon', 'search_connection']:
        cfg[section].update(user_cfg.get(section, {}))

    for section in ['preferred_hosts', 'ignore_hosts']:
        cfg[section] = user_cfg.get(section, [])

    return cfg
