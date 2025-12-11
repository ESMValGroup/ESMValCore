"""esgf-pyclient configuration.

The configuration is read from the file ~/.esmvaltool/esgf-pyclient.yml.
"""

import logging
import os
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".esmvaltool" / "data-intake.yml"


def read_config_file() -> dict:
    """Read the configuration from file."""
    if CONFIG_FILE.exists():
        logger.info("Loading Intake-ESM configuration from %s", CONFIG_FILE)
        mode = os.stat(CONFIG_FILE).st_mode
        if mode & stat.S_IRWXG or mode & stat.S_IRWXO:
            logger.warning("Correcting unsafe permissions on %s", CONFIG_FILE)
            os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
        with CONFIG_FILE.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    else:
        logger.info(
            "Using default Intake-ESM configuration, configuration "
            "file %s not present.",
            CONFIG_FILE,
        )
        cfg = {}

    return cfg


def load_intake_config():
    """Load the intake-esm configuration."""
    cfg = {
        "CMIP6": {
            "catalogs": {
                "NCI": [
                    {
                        "file": "/g/data/fs38/catalog/v2/esm/catalog.json",
                        "facets": {
                            "activity": "activity_id",
                            "dataset": "source_id",
                            "ensemble": "member_id",
                            "exp": "experiment_id",
                            "grid": "grid_label",
                            "institute": "institution_id",
                            "mip": "table_id",
                            "short_name": "variable_id",
                            "version": "version",
                            "frequency": "frequency",
                        },
                    },
                    {
                        "file": "/g/data/oi10/catalog/v2/esm/catalog.json",
                        "facets": {
                            "activity": "activity_id",
                            "dataset": "source_id",
                            "ensemble": "member_id",
                            "exp": "experiment_id",
                            "grid": "grid_label",
                            "institute": "institution_id",
                            "mip": "table_id",
                            "short_name": "variable_id",
                            "version": "version",
                            "frequency": "frequency",
                        },
                    },
                ]
            }
        },
        "CMIP5": {
            "catalogs": {
                "NCI": [
                    {
                        "file": "/g/data/rr3/catalog/v2/esm/catalog.json",
                        "facets": {
                            "activity": "activity_id",
                            "dataset": "source_id",
                            "ensemble": "ensemble",
                            "exp": "experiment",
                            "grid": "grid_label",
                            "institute": "institution_id",
                            "mip": "table_id",
                            "short_name": "variable",
                            "version": "version",
                        },
                    },
                    {
                        "file": "/g/data/al33/catalog/v2/esm/catalog.json",
                        "facets": {
                            "activity": "activity_id",
                            "dataset": "source_id",
                            "ensemble": "ensemble",
                            "exp": "experiment",
                            "institute": "institute",
                            "mip": "table",
                            "short_name": "variable",
                            "version": "version",
                            "timerange": "time_range",
                        },
                    },
                ]
            }
        },
    }

    file_cfg = read_config_file()
    cfg.update(file_cfg)

    return cfg


@lru_cache()
def get_intake_config():
    """Get the esgf-pyclient configuration."""
    return load_intake_config()


def _read_facets(
    cfg: dict,
    fhandle: str | None,
    project: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Extract facet mapping from ESMValCore configuration for a given catalog file handle.

    Recursively traverses the ESMValCore configuration structure to find the
    facet mapping that corresponds to the specified file handle.

    Parameters
    ----------
    cfg : dict
        The ESMValCore intake configuration dictionary.
    fhandle : str
        The file handle/path of the intake-esm catalog to match.
    project : str, optional
        The current project name in the configuration hierarchy.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: Facet mapping between ESMValCore facets and catalog columns
        - str: The project name associated with the catalog file
    """
    if fhandle is None:
        raise ValueError(
            "Unable to ascertain facets without valid file handle."
        )

    for _project, val in cfg.items():
        if not (isinstance(val, list)):
            return _read_facets(val, fhandle, project or _project)
        for facet_info in val:
            file, facets = facet_info.get("file"), facet_info.get("facets")
            if file == fhandle:
                return facets, project  # type: ignore[return-value]
    else:
        raise ValueError(
            f"No facets found for {fhandle} in the config file. "
            "Please check the config file and ensure it is valid."
        )
