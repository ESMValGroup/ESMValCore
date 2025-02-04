import logging
from pathlib import Path

import intake
import intake_esm

from ..config._config import get_project_config
from ..local import DataSource, _select_files

__all__ = ["IntakeDataSource", "get_project_config", "_select_files"]

logger = logging.getLogger(__name__)

_CACHE: dict[Path, intake_esm.core.esm_datastore] = {}


def clear_catalog_cache():
    """Clear the catalog cache."""
    _CACHE.clear()


def load_catalog(project, drs):
    """Load an intake-esm catalog and associated facet mapping."""
    catalog_info = get_project_config(project).get("catalogs", {})
    site = drs.get(project, "default")
    if site not in catalog_info:
        return None, {}

    catalog_url = Path(catalog_info[site]["file"]).expanduser()

    if catalog_url not in _CACHE:
        logger.info(
            "Loading intake-esm catalog (this may take some time): %s",
            catalog_url,
        )
        _CACHE[catalog_url] = intake.open_esm_datastore(catalog_url)
        logger.info("Done loading catalog")

    catalog = _CACHE[catalog_url]
    facets = catalog_info[site]["facets"]
    return catalog, facets


def find_files(variable, drs):
    """Find files for variable in intake-esm catalog."""
    catalog, facets = load_catalog(variable["project"], drs)
    if not catalog:
        return []

    query = {}
    for ours, theirs in facets.items():
        if ours == "version" and "version" not in variable:
            # skip version if not specified in recipe
            continue
        query[theirs] = variable[ours]

    selection = catalog.search(**query)

    # Select latest version
    if "version" not in variable and "version" in facets:
        latest_version = selection.df[facets["version"]].max()
        variable["version"] = latest_version
        query = {
            facets["version"]: latest_version,
        }
        selection = selection.search(**query)

    filenames = list(selection.df["path"])

    # Select only files within the time range
    filenames = _select_files(
        filenames, variable["start_year"], variable["end_year"]
    )

    return filenames


class IntakeDataSource(DataSource):
    """
    Class to handle loading data using Intake-esm.
    """

    def __init__():
        pass
