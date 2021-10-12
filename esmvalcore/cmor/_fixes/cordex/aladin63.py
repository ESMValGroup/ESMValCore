"""Fixes for ALADIN63 model."""
from ..fix import Fix


class tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """
        Fixes incorrect units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)

        # some files have height of 10m not 2m
        cube.coord("height").points = 2

        return cubes
