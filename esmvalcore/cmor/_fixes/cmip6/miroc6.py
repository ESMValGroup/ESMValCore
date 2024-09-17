"""Fixes for MIROC6 model."""
import numpy as np

from ..common import ClFixHybridPressureCoord
from ..fix import Fix

Cl = ClFixHybridPressureCoord

Cli = ClFixHybridPressureCoord

Clw = ClFixHybridPressureCoord


class Tos(Fix):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """
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
            for coord_name in ['latitude', 'longitude']:
                coord = cube.coord(coord_name)
                coord.points = coord.core_points().astype(np.float32).astype(
                    np.float64)
                if not coord.has_bounds():
                    coord.guess_bounds()
                coord.bounds = coord.core_bounds().astype(np.float32).astype(
                    np.float64)
        return cubes
