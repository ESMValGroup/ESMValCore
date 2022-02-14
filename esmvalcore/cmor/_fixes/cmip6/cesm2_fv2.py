"""Fixes for CESM2-FV2 model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Tas as BaseTas
from ..fix import Fix
from ..shared import fix_ocean_depth_coord
import numpy as np
import cf_units
from ..common import SiconcFixScalarCoord


Cl = BaseCl


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Siconc = SiconcFixScalarCoord


Tas = BaseTas


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
            if cube.coords('latitude'):
                cube.coord('latitude').var_name = 'lat'
            if cube.coords('longitude'):
                cube.coord('longitude').var_name = 'lon'

            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if str(z_coord.units).lower() in ['cm', 'centimeters'] and np.max(z_coord.points)>10000.:
                    z_coord.units = cf_units.Unit('m')
                    z_coord.points = z_coord.points /100.
                if str(z_coord.units).lower() in ['cm', 'centimeters'] and np.max(z_coord.points)<10000.:
                    z_coord.units = cf_units.Unit('m')
                    #z_coord.points = z_coord.points /100.


                #z_coord = cube.coord(axis='Z')
                #if z_coord.var_name == 'olevel':
                fix_ocean_depth_coord(cube)
        return cubes

