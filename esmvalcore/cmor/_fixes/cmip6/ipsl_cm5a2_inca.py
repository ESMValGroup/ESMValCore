"""Fixes for IPSL-CM5A2-INCA model."""

from esmvalcore.cmor._fixes.fix import Fix

from .ipsl_cm6a_lr import AllVars as BaseAllVars
from .ipsl_cm6a_lr import Clcalipso as BaseClcalipso
from .ipsl_cm6a_lr import Omon as BaseOmon

AllVars = BaseAllVars


Clcalipso = BaseClcalipso


Omon = BaseOmon

class Snw(Fix):
    """Fixes for ``snw``."""

    def fix_metadata(self, cubes):
        """Fix ``lon`` coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList

        """
        
        for cube in cubes:
            coord_names = [cor.standard_name for cor in cube.coords()]
            if "longitude" in coord_names:
                lon_coord = cube.coord("longitude")
                if (lon_coord.points[0] == 360.0):
                    lon_coord.points[0] = 0.0

        return cubes
