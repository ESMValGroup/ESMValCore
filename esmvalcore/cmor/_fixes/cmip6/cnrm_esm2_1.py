"""Fixes for CNRM-ESM2-1 model."""
from ..fix import Fix
from ..shared import (fix_ocean_depth_coord)


from .cnrm_cm6_1 import Cl as BaseCl
from .cnrm_cm6_1 import Clcalipso as BaseClcalipso
from .cnrm_cm6_1 import Cli as BaseCli
from .cnrm_cm6_1 import Clw as BaseClw


Cl = BaseCl


Clcalipso = BaseClcalipso


Cli = BaseCli


Clw = BaseClw


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
                if z_coord.standard_name is None:
                    fix_ocean_depth_coord(cube)
        return cubes
