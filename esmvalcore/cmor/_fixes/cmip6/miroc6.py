"""Fixes for MIROC6 model."""
from ..cmip5.bcc_csm1_1 import Cl as BaseCl


class Cl(BaseCl):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Remove attributes from ``Surface Air Pressure`` coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cubes = super().fix_metadata(cubes)
        cube = self.get_cube_from_list(cubes)
        coord = cube.coord(long_name='Surface Air Pressure')
        coord.attributes = {}
        return cubes


class Cli(Cl):
    """Fixes for ``cli``."""


class Clw(Cl):
    """Fixes for ``clw``."""
