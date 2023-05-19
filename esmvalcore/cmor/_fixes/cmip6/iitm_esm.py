"""Fixes for IITM-ESM model."""
from ..common import OceanFixGrid
from ..fix import Fix
import numpy as np
from esmvalcore.cmor.check import _get_time_bounds

Tos = OceanFixGrid
# Omon = OceanFixGrid # CHeck 

class AllVars(Fix):
    """Fixes for all vars."""
    def fix_metadata(self, cubes):
        """Fix parent time units.

        IITM-ESM monthly data may have a bad time bounds spanning 20 days.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            freq = self.extra_facets["frequency"]
            time = cube.coord("time", dim_coords=True)
            bounds = _get_time_bounds(time, freq)
            if np.any(bounds != time.bounds):
                time.bounds = bounds
        return cubes