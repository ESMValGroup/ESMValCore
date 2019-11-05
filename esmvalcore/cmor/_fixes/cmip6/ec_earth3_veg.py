"""Fixes for EC-Earth3-Veg."""
from ..fix import Fix
import cf_units

class msftyz(Fix):
    """Fix msftyz."""

    def fix_metadata(self, cubes):
        """
        Fix standard and long name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            basin = cube.coord('region')
            basin.var_name = 'basin'
        return cubes
