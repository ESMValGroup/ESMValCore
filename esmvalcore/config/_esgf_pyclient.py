"""esgf-pyclient configuration.

The configuration is read from the file ~/.esmvaltool/esgf-pyclient.yml.
"""

import logging
import os
import stat
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".esmvaltool" / "esgf-pyclient.yml"


def read_config_file():
    """Read the configuration from file."""
    if CONFIG_FILE.exists():
        logger.info("Loading ESGF configuration from %s", CONFIG_FILE)
        mode = os.stat(CONFIG_FILE).st_mode
        if mode & stat.S_IRWXG or mode & stat.S_IRWXO:
            logger.warning("Correcting unsafe permissions on %s", CONFIG_FILE)
            os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
        with CONFIG_FILE.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    else:
        logger.info(
            "Using default ESGF configuration, configuration "
            "file %s not present.",
            CONFIG_FILE,
        )
        cfg = {}

    # For backwards compatibility: prior to v2.6 the configuration file
    # contained a single URL instead of a list of URLs.
    if "search_connection" in cfg:
        if "url" in cfg["search_connection"]:
            url = cfg["search_connection"].pop("url")
            cfg["search_connection"]["urls"] = [url]

    return cfg


def load_esgf_pyclient_config():
    """Load the esgf-pyclient configuration."""
    cfg = {
        # Arguments to
        # https://esgf-pyclient.readthedocs.io/en/latest/api.html#pyesgf.search.connection.SearchConnection
        "search_connection": {
            # Be careful about the url, not all search urls have CMIP3 data?
            "urls": [
                # Use "https://esgf-node.ornl.gov/esgf-1-5-bridge" once the Solr
                # based indices below are no longer available.
                # See https://github.com/ESMValGroup/ESMValCore/issues/2757 and
                # linked issues and pull requests for additional information.
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
            "cache": "~/.esmvaltool/cache/pyesgf-search-results",
            "expire_after": 86400,  # cache expires after 1 day
        },
    }

    file_cfg = read_config_file()
    for section in ["search_connection"]:
        cfg[section].update(file_cfg.get(section, {}))

    if "cache" in cfg["search_connection"]:
        cache_file = (
            Path(os.path.expandvars(cfg["search_connection"]["cache"]))
            .expanduser()
            .absolute()
        )
        cfg["search_connection"]["cache"] = cache_file
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    return cfg


@lru_cache
def get_esgf_config():
    """Get the esgf-pyclient configuration."""
    return load_esgf_pyclient_config()
