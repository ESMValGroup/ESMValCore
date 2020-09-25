"""Fixes for CMIP6 HadGEM-GC31-LL."""
from ..common import ClFixHybridHeightCoord
from .ukesm1_0_ll import AllVars as BaseAllVars
from ..fix import Fix


AllVars = BaseAllVars


Cl = ClFixHybridHeightCoord


Cli = ClFixHybridHeightCoord


Clw = ClFixHybridHeightCoord


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
            print('\nbasin:', basin)

        return cubes
