# height 2m a differents altures
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord

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
            add_scalar_height_coord(cube)
            if cube.coord('height').points != 2.:
                cube.coord('height').points = np.ma.array([2.0])
            cube.coord('time').long_name = 'time'

        return cubes

class Pr(Fix):
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
            cube.coord('time').long_name = 'time'

        return cubes