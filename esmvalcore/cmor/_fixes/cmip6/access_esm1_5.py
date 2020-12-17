"""Fixes for ACCESS-ESM1-5."""
from ..common import ClFixHybridHeightCoord
from ..fix import Fix
import numpy as np


class Zg(Fix):
    """Fixes for zg."""
    def fix_metadata(self, cubes):
        """Correctly round air pressure coordinate.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.coord('air_pressure').points = \
            np.round(cube.coord('air_pressure').points, 0)
        cube.coord('air_pressure').bounds = \
            np.round(cube.coord('air_pressure').bounds, 0)
        return cubes

class Hus(Fix):
    """Fixes for hus."""
    def fix_metadata(self, cubes):
        """Correctly round air pressure coordinate.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.coord('air_pressure').points = \
            np.round(cube.coord('air_pressure').points, 0)
        cube.coord('air_pressure').bounds = \
            np.round(cube.coord('air_pressure').bounds, 0)
        return cubes

Cl = ClFixHybridHeightCoord


Cli = ClFixHybridHeightCoord


Clw = ClFixHybridHeightCoord
