"""Fixes for CMIP6 UKESM1-0-LL."""
from .hadgem3_gc31_ll import AllVars as BaseAllVars
from ..fix import Fix


class AllVars(BaseAllVars):
    """Fixes for all vars."""

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
