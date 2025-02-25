import logging
import datetime
from pathlib import Path
from typing import Sequence
import isodate

import intake
import intake_esm
import iris
from ncdata.iris_xarray import cubes_from_xarray

from esmvalcore.config._config import get_project_config
from esmvalcore.config import CFG, Config, Session
from esmvalcore.preprocessor import clip_timerange, fix_metadata
from esmvalcore.cmor.table import VariableInfo
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
            self._files = self._find_files(self.facets, CFG["drs"])
        return self._files

    def load(self) -> iris.cube.Cube:
        """
        Load the dataset. We should be able to do this by just querying for catalog
        for the files in self.files and then loading them into an xarray dataset.
        This isthen converted to an iris cube.
        """
        cat_subset = self.catalog.search(path=self.files)
        if len(cat_subset) > 1:
            raise ValueError("Multiple datasets found for the same query")
        elif len(cat_subset) == 0:
            raise ValueError("No datasets found for the query")

        ds = cat_subset.to_dask()
        da = ds[self.facets["short_name"]]
        cube = da.to_iris()

        extra_facets = {k : v for k, v in self.facets.items() if k not in ["short_name", "project", "dataset","mip","frequency"]}
        cube = fix_metadata((cube,), 
                            short_name = self.facets["short_name"],
                            project = self.facets["project"],
                            dataset = self.facets["dataset"],
                            mip = self.facets["mip"],
                            frequency = self.facets.get("frequency"),
                            **extra_facets
        )[0]

        return clip_timerange(cube, self.facets["timerange"])


    def _find_files(self, variable : dict , drs) -> Sequence[File]:
        """Find files for variable in intake-esm catalog.

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

        return selection.unique().path
