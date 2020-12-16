"""Fixes for KIOST-ESM model."""
import iris

from ..fix import Fix
from ..shared import add_scalar_height_coord

class Tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return [cube]
