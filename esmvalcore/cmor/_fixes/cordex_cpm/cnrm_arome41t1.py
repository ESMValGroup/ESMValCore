"""Fixes for CNRM-AROME41t1 model."""
from ..fix import Fix

import iris


class tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """
        Add height coordinate

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)

        height_coord = iris.coords.AuxCoord(
            2, units='m',
            standard_name='height', var_name='height', long_name='height2m'
            )

        cube.add_aux_coord(height_coord)

        return cubes
