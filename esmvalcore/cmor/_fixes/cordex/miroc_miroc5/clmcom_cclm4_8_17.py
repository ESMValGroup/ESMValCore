"""Fixes for rcm CLMcom-CCLM4-8-17 driven by MIROC-MIROC5."""
from esmvalcore.cmor.fix import Fix

from cf_units import Unit
import numpy as np

class AllVars(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Set calendar to 'proleptic_gregorian' to avoid
        concatenation issues between historical and
        scenario runs.

        Fix dtype value of coordinates and coordinate bounds.

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
        
        # Further issues appear, maybe because historical data lat/lon 
        # does not have bounds whereas scenario data has them.
                    

        return cubes