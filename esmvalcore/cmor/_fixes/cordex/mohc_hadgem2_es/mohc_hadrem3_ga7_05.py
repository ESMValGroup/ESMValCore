
from esmvalcore.cmor.fix import Fix

import numpy as np

class Tas(Fix):
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
          cube.coord('latitude').var_name = 'lat'
          cube.coord('longitude').var_name = 'lon'
          cube.coord('time').long_name = 'time'

        return cubes

Pr = Tas