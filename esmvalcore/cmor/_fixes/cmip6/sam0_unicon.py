"""Fixes for SAM0-UNICON model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


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
