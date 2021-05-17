"""Fixes for GFDL-ESM4 model."""
import iris
from ..fix import Fix
from ..shared import add_scalar_typesi_coord


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
