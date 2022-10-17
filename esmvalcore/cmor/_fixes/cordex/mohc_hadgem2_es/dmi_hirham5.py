
from esmvalcore.cmor.fix import Fix

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
          cube.coord('latitude').attributes = {}
          cube.coord('longitude').attributes = {}

        return cubes