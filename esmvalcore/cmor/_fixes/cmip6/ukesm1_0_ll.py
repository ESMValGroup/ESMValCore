"""Fixes for CMIP6 UKESM1-0-LL."""
from ..common import ClFixHybridHeightCoord
from ..fix import Fix


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        parent_units = 'parent_time_units'
        bad_value = 'days since 1850-01-01-00-00-00'
        for cube in cubes:
            try:
                if cube.attributes[parent_units] == bad_value:
                    cube.attributes[parent_units] = 'days since 1850-01-01'
            except AttributeError:
                pass
        return cubes


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

Cl = ClFixHybridHeightCoord


Cli = ClFixHybridHeightCoord


Clw = ClFixHybridHeightCoord
