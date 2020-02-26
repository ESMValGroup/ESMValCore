
"""Fix HadGEM2_ES."""
import numpy as np

from ..fix import Fix


class AllVars(Fix):
    """Fix errors common to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix latitude.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            lats = cube.coords('latitude')
            if lats:
                lat = cube.coord('latitude')
                lat.points = np.clip(lat.points, -90., 90.)
                lat.bounds = np.clip(lat.bounds, -90., 90.)

        return cubes


class O2(Fix):
    """Fix o2."""

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
        std = 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water'
        long_name = 'Dissolved Oxygen Concentration'

        cubes[0].long_name = long_name
        cubes[0].standard_name = std
        return cubes
