# height 2m a differents altures
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord

from cf_units import Unit
import iris
import numpy as np

class AllVars(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate.

        Fix also done for prw.
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coord('time').units.calendar == 'gregorian':
                cube.coord('time').units = Unit(
                    'days since 1850-1-1 00:00:00',
                    calendar='proleptic_gregorian'
                )
            for coord in cube.coords():
                if coord.dtype in ['>f8', '>f4']:
                    coord.points = coord.core_points().astype(
                        np.float64, casting='same_kind')
                    if coord.bounds is not None:
                        coord.bounds = coord.core_bounds().astype(
                            np.float64, casting='same_kind')
        # further issues with dtype, may be due because historical data lat/lon does not have bounds whereas scenario data has them
                    

        return cubes