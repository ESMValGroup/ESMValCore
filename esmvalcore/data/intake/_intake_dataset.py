"""Import datasets using Intake-ESM."""

import logging
from pathlib import Path
from typing import Any, Sequence

import intake
import intake_esm

from esmvalcore.config._intake import get_intake_config
from esmvalcore.local import LocalFile

__all__ = ["load_catalogs", "clear_catalog_cache"]

logger = logging.getLogger(__name__)

_CACHE: dict[Path, intake_esm.core.esm_datastore] = {}


def clear_catalog_cache() -> None:
    """Clear the catalog cache."""
    _CACHE.clear()


def load_catalogs(
    project: str, drs: dict[str, Any]
) -> tuple[list[intake_esm.core.esm_datastore], list[dict[str, str]]]:
    """Load all intake-esm catalogs for a project and their associated facet mappings.

    Parameters
    ----------
    project : str
        The project name, eg. 'CMIP6'.
    drs : dict
        The DRS configuration. Can be obtained from the global configuration drs
        field, eg. CFG['drs'].

    Returns
    -------
    intake_esm.core.esm_datastore
        The catalog.
    dict
        The facet mapping - a dictionary mapping ESMVlCore dataset facet names
        to the fields in the intake-esm datastore.
    """
    catalog_info: dict[str, Any] = (
        get_intake_config().get(project, {}).get("catalogs", {})
    )

    site = drs.get(project, "default")
    if site not in catalog_info:
        return [None], [{}]

    catalog_urls = [
        Path(catalog.get("file")).expanduser()
        for catalog in catalog_info[site]
    ]
    facet_list = [catalog.get("facets") for catalog in catalog_info[site]]

    for catalog_url in catalog_urls:
        if catalog_url not in _CACHE:
            logger.info(
                "Loading intake-esm catalog (this may take some time): %s",
                catalog_url,
            )
            _CACHE[catalog_url] = intake.open_esm_datastore(catalog_url)
            logger.info("Successfully loaded catalog %s", catalog_url)

    return ([_CACHE[cat_url] for cat_url in catalog_urls], facet_list)


def find_files(
    *, project: str, drs: dict, facets: dict
) -> Sequence[LocalFile]:
    """Find files for variable in all intake-esm catalogs associated with a project.

    Parameters
    ----------
    facet_map : dict
        A dict mapping the variable names used to initialise the IntakeDataset
        object to their ESMValCore facet names. For example,
        ```
        ACCESS_ESM1_5 = IntakeDataset(
            short_name='tos',
            project='CMIP6',
        )
        ```
        would result in a variable dict of {'short_name': 'tos', 'project': 'CMIP6'}.
    drs : dict
        The DRS configuration. Can be obtained from the global configuration drs
        field, eg. CFG['drs'].
    """
    catalogs, facet_maps = load_catalogs(project, drs)

    if not catalogs:
        return []

    files = []

    for catalog, facet_map in zip(catalogs, facet_maps, strict=False):
        query = {facet_map.get(key): val for key, val in facets.items()}
        query.pop(None, None)

        _unused_facets = {
            key: val for key, val in facets.items() if key not in facet_map
        }

        logger.info(
            "Unable to refine datastore search on catalog %s with the following facets %s",
            catalog.esmcat.catalog_file,
            _unused_facets,
        )

        selection = catalog.search(**query)

        if not selection:
            continue

        # Select latest version
        if "version" in facet_map and "version" not in facets:
            latest_version = max(
                selection.unique().version
            )  # These are strings - need to double check the sorting here.
            query = {
                **query,
                facet_map["version"]: latest_version,
            }
            selection = selection.search(**query)

            files += [LocalFile(f) for f in selection.unique().path]

    return files
