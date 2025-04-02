"""Import datasets using Intake-ESM."""

import logging
from numbers import Number
from pathlib import Path
from typing import Any, Sequence

import intake
import intake_esm

from esmvalcore.config import CFG
from esmvalcore.config._config import get_project_config
from esmvalcore.dataset import Dataset, File
from esmvalcore.local import LocalFile
from esmvalcore.typing import Facets

__all__ = ["IntakeDataset", "load_catalogs", "clear_catalog_cache"]

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
    catalog_info: dict[str, Any] = get_project_config(project).get(
        "catalogs", {}
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


class IntakeDataset(Dataset):
    """Class to handle loading data using Intake-ESM.

    Crucially, we do not subclass Dataset, as this is going to cause problems.
    """

    def __init__(self, **facets: dict[str, Any]) -> None:
        project: str = facets["project"]  # type: ignore[assignment]
        self.facets: Facets = {}
        self.catalog, self._facets = load_catalogs(project, CFG["drs"])
        self._unmapped_facets: dict[str, Any] = {}
        self._files: Sequence[File] | None = None

    @property
    def files(self) -> Sequence[File]:
        if self._files is None:
            self._files = self._find_files(self.facets, CFG["drs"])
        return self._files

    @files.setter
    def files(self, value: Sequence[File]):
        """Manually set the files for the dataset."""
        self._files = value

    @property
    def filenames(self) -> Sequence[str]:
        """String representation of the filenames in the dataset."""
        return [str(f) for f in self.files]

    def _find_files(  # type: ignore[override]
        self,
        facet_map: dict[str, str | Sequence[str] | Number],
        drs: dict[str, Any],
    ) -> Sequence[File]:
        """Find files for variable in all intake-esm catalogs associated with a project.

        As a side effect, sets the unmapped_facets attribute - this is used to
        cache facets which are not in the datastore.

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
        if not isinstance(facet_map["project"], str):
            raise TypeError(
                "The project facet must be a string for Intake Datasets."
            )

        catalogs, facets_list = load_catalogs(facet_map["project"], drs)
        if not catalogs:
            return []

        files = []

        for catalog, facets in zip(catalogs, facets_list, strict=False):
            query = {val: facet_map.get(key) for key, val in facets.items()}
            query = {key: val for key, val in query.items() if val is not None}

            unmapped = {
                key: val for key, val in facet_map.items() if key not in facets
            }
            unmapped.pop("project", None)

            self._unmapped_facets = unmapped

            selection = catalog.search(**query)

            # Select latest version
            if "version" in facets and "version" not in facet_map:
                latest_version = max(
                    selection.unique().version
                )  # These are strings - need to double check the sorting here.
                facet_map["version"] = latest_version
                query = {
                    facets["version"]: latest_version,
                }
                selection = selection.search(**query)

                files += [LocalFile(f) for f in selection.unique().path]

        self.augment_facets()
        return files


"""
def find_files(*, project, short_name, dataset, **facets):
    catalog, facet_map = load_catalogs(project, CFG["drs"])

    if not isinstance(facet_map["project"], str):
        raise TypeError(
            "The project facet must be a string for Intake Datasets."
        )

    # catalogs, facets_list = load_catalogs(facet_map["project"], drs)
    if not catalogs:
        return []

    files = []

    for catalog, facets in zip(catalogs, facets_list, strict=False):
        query = {val: facet_map.get(key) for key, val in facets.items()}
        query = {key: val for key, val in query.items() if val is not None}

        unmapped = {
            key: val for key, val in facet_map.items() if key not in facets
        }
        unmapped.pop("project", None)

        # self._unmapped_facets = unmapped

        selection = catalog.search(**query)

        # Select latest version
        if "version" in facets and "version" not in facet_map:
            latest_version = max(
                selection.unique().version
            )  # These are strings - need to double check the sorting here.
            facet_map["version"] = latest_version
            query = {
                facets["version"]: latest_version,
            }
            selection = selection.search(**query)

            files += [LocalFile(f) for f in selection.unique().path]

    # self.augment_facets()
    return files

"""
