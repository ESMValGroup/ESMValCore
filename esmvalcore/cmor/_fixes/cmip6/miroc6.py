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
        """Fix latitude_bounds and longitude_bounds data type and round to 4
        d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            latitude = cube.coord('latitude')
            if latitude.bounds is None:
                latitude.guess_bounds()
            latitude.bounds = latitude.bounds.astype(np.float32)
            latitude.bounds = np.round(latitude.bounds, 4)
            latitude.points = latitude.points.astype(np.float32)
            longitude = cube.coord('longitude')
            if longitude.bounds is None:
                longitude.guess_bounds()
            longitude.bounds = longitude.bounds.astype(np.float32)
            longitude.bounds = np.round(longitude.bounds, 4)
            longitude.points = longitude.points.astype(np.float32)
        return cubes
