"""Fixes for E3SM-1-0."""
from .fix import Fix
from numpy import array 

class msftmz(Fix):
    """Fix msftmz."""

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
            basin.points = array(['global_ocean', 'atlantic_arctic_ocean', ])
        return cubes

