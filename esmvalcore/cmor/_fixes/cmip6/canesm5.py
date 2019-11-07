"""Fixes for CMIP6 CanESM5."""
from ..fix import Fix


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

        return cubes
