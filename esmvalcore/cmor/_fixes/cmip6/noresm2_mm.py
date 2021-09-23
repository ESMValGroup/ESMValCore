"""Fixes for NorESM2-MM model."""
from ..common import ClFixHybridPressureCoord

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord

class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if z_coord.var_name == 'olevel':
                    fix_ocean_depth_coord(cube)
        return cubes


