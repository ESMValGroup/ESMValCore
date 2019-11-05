"""Fixes for CNRM-CM6-1."""
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
            gridlat = cube.coord('Ocean grid longitude mean')
            gridlat.var_name = 'rlat'
            gridlat.standard_name='grid_latitude'
            gridlat.units=cf_units.Unit('degrees')
            gridlat.long_name='Grid Latitude'
            print(gridlat.points)
            # These values are wrong - they are supposed to be latitude
            # values but they are actually y axis indices.

            # basin = cube.coord('region')
            # basin.var_name = 'basin'
         assert 0
        return cubes
