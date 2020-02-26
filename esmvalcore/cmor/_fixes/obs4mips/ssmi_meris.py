
"""Fixes for CCSM4 model."""
from iris.cube import CubeList

from ..fix import Fix


class Prw(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Remove error and number of observations cubes

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        return CubeList([cube])
