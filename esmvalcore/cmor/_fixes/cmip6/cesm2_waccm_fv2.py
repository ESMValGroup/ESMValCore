"""Fixes for cesm2-waccm-fv2."""
from iris.cube import CubeList
"""Fixes for CESM2-WACCM-FV2 model."""
from .cesm2 import Tas as BaseTas
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw
from ..common import SiconcFixScalarCoord

from ..fix import Fix
from ..shared import fix_ocean_depth_coord

import numpy as np
import cf_units

class AllVars(Fix):
    """Fixes for thetao."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.coords('latitude'):
            cube.coord('latitude').var_name = 'lat'
        if cube.coords('longitude'):
            cube.coord('longitude').var_name = 'lon'
        return CubeList([cube])


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if str(z_coord.units).lower() in ['cm', 'centimeters'] and np.max(z_coord.points)>10000.:
                    z_coord.units = cf_units.Unit('m')
                    z_coord.points = z_coord.points /100.
                if str(z_coord.units).lower() in ['cm', 'centimeters'] and np.max(z_coord.points)<10000.:
                    z_coord.units = cf_units.Unit('m')
#                    z_coord.points = z_coord.points /100.

                fix_ocean_depth_coord(cube)
        return cubes

Fgco2 = BaseFgco2


Siconc = SiconcFixScalarCoord


Tas = BaseTas
