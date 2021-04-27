"""Fixes for CNRM-ESM2-1 model."""
from iris.cube import CubeList

from .cnrm_cm6_1 import Cl as BaseCl
from .cnrm_cm6_1 import Clcalipso as BaseClcalipso
from .cnrm_cm6_1 import Cli as BaseCli
from .cnrm_cm6_1 import Clw as BaseClw

from ..fix import Fix
from ..shared import (set_ocean_depth_coord)


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """
        Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        new_list = CubeList()
        for cube in cubes:
            if cube.coords(axis='Z'):
                if not cube.coord(axis='Z').standard_name:
                    cube = set_ocean_depth_coord(cube)
            new_list.append(cube)
        return CubeList(new_list)


Cl = BaseCl


Clcalipso = BaseClcalipso


Cli = BaseCli


Clw = BaseClw
