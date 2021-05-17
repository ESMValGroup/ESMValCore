"""Fixes for CESM2-WACCM-FV2 model."""
from .cesm2 import Tas as BaseTas
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw
from ..fix import Fix
from ..shared import add_scalar_typesi_coord


Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


class Siconc(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """Add typesi coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typesi_coord(cube)
        return cubes


Tas = BaseTas
