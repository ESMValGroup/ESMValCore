import logging
from pathlib import Path
from typing import Sequence
# import isodate

import intake
import intake_esm
import iris
# from ncdata.iris_xarray import cubes_from_xarray

from esmvalcore.config._config import get_project_config
from esmvalcore.config import CFG
from esmvalcore.preprocessor import clip_timerange, fix_metadata
# from esmvalcore.cmor.table import VariableInfo
from esmvalcore.dataset import Dataset, File
from esmvalcore.local import LocalFile
# from esmvalcore.typing import Facets, FacetValue

__all__ = ["IntakeDataset", "get_project_config", "_select_files"]

logger = logging.getLogger(__name__)

_CACHE: dict[Path, intake_esm.core.esm_datastore] = {}



def clear_catalog_cache():
    """Clear the catalog cache."""
    _CACHE.clear()


def load_catalogs(project, drs) -> tuple[list[intake_esm.core.esm_datastore], list[dict]]:
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
    catalog_info = get_project_config(project).get("catalogs", {})
    site = drs.get(project, "default")
    if site not in catalog_info:
        return None, {}

    catalog_urls = [
     Path(catalog.get('file')).expanduser() for catalog in catalog_info[site]
    ]
    facet_list = [
        catalog.get('facets') for catalog in catalog_info[site]
    ]

    for catalog_url in catalog_urls:
        if catalog_url not in _CACHE:
            logger.info(
                "Loading intake-esm catalog (this may take some time): %s",
                catalog_url,
            )
            _CACHE[catalog_url] = intake.open_esm_datastore(catalog_url)
            logger.info("Successfully loaded catalog %s", catalog_url)

    return (
         [_CACHE[cat_url] for cat_url in catalog_urls], facet_list
    )



class IntakeDataset(Dataset):
    """
    Class to handle loading data using Intake-esm.
    """

    def __init__(self,  **facets):
        project = facets["project"]
        self.catalog, self._facets = load_catalogs(project, CFG["drs"])
        super().__init__(**facets)

    @property
    def files(self) -> Sequence[File]:
        if self._files is None:
            self._files = self._find_files(self.facets, CFG["drs"])
        return self._files

    @property
    def filenames(self) -> Sequence[str]:
        return [str(f) for f in self.files]

    def _find_files(self, variable : dict , drs) -> Sequence[File]:
        """Find files for variable in all intake-esm catalogs associated with a
        project.

        As a side effect, sets the unmapped_facets attribute - this is used to
        cache facets which are not in the datastore.


        Parameters
        ----------
        variable : dict
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
        catalogs, facets_list = load_catalogs(variable["project"], drs)
        if not catalogs:
            return []

        files = []

        for catalog, facets in zip(catalogs, facets_list):
            query = {val : variable.get(key) for key, val in facets.items()}
            query = {key : val for key, val in query.items() if val is not None}

            unmapped = {key: val for key, val in variable.items() if key not in facets}
            unmapped.pop("project", None)

            self._unmapped_facets = unmapped

            selection = catalog.search(**query)

            # Select latest version
            if 'version' in facets and 'version' not in variable:
                latest_version = max(selection.unique().version) # These are strings - so wtf doe this mean?
                variable['version'] = latest_version
                query = {
                    facets['version']: latest_version,
                }
                selection = selection.search(**query)

                files += [LocalFile(f) for f in selection.unique().path]


        self.augment_facets()
        return files
