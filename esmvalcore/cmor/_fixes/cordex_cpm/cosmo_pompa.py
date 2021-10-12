"""Fixes for COSMO-pompa model."""
from ..fix import Fix


class tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """
        Fix height coordinate

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)

        cube.coord('height').var_name = 'height'

        return cubes


class pr(Fix):
    """Fixes for pr."""
    def fix_metadata(self, cubes):
        """
        Fix height coordinate

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)

        # correct units
        cube.units = 'kg m-2 s-1'

        return cubes
