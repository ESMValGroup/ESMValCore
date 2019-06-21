"""Fixes for CESM2-WACCM model."""
from ..fix import Fix
from ..shared import add_height_coord


class tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        add_height_coord(cube)
        return cubes
