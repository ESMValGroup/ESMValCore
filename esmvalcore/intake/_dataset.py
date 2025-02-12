import logging
from pathlib import Path
from typing import Sequence

import intake
import intake_esm

from esmvalcore.config._config import get_project_config
from esmvalcore.config import CFG, Config, Session
from ..cmor.table import VariableInfo
from esmvalcore.dataset import Dataset, File
from esmvalcore.typing import Facets, FacetValue

__all__ = ["IntakeDataset", "get_project_config", "_select_files"]

logger = logging.getLogger(__name__)

_CACHE: dict[Path, intake_esm.core.esm_datastore] = {}



def clear_catalog_cache():
    """Clear the catalog cache."""
    _CACHE.clear()


def load_catalog(project, drs) -> tuple[intake_esm.core.esm_datastore, dict]:
    """Load an intake-esm catalog and associated facet mapping.

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
        The facet mapping - a dictionarry mapping ESMVlCore dataset facet names 
        to the fields in the intake-esm datastore.
    """
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


def find_files(variable : dict , drs):
    """Find files for variable in intake-esm catalog.

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
    catalog, facets = load_catalog(variable["project"], drs)
    if not catalog:
        return []

    query = {val : variable.get(key) for key, val in facets.items()}
    query = {key : val for key, val in query.items() if val is not None}
    """
    for ours, theirs in facets.items():
        if ours == 'version' and 'version' not in variable:
            # skip version if not specified in recipe
            continue
        query[theirs] = variable[ours]

    ^ Check this logic is retained
    """

    unmapped = {key: val for key, val in variable.items() if key not in facets}
    unmapped.pop("project", None)

    selection = catalog.search(**query)

    # Select latest version
    if 'version' in facets and 'version' not in variable:
        latest_version = selection.df[facets['version']].max() # These are strings - so wtf doe this mean?
        variable['version'] = latest_version
        query = {
            facets['version']: latest_version,
        }
        selection = selection.search(**query)

    # Can we do this without all the instantiations into a dataframe? I think so
    return selection.df['path'].tolist()



class IntakeDataset(Dataset):
    """
    Class to handle loading data using Intake-esm.
    """

    def __init__(self,  **facets):
        project = facets["project"]
        self.catalog, self._facets = load_catalog(project, CFG["drs"])
        super().__init__(**facets)

    @property
    def files(self) -> Sequence[File]:
        if self._files is None:
            self._files = find_files(self.facets, CFG["drs"])
        return self._files

    def load():
        """
        Load the dataset.
        """
        super().load()

    


        

