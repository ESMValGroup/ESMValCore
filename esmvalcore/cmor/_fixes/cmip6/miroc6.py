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
            latitude = cube.coord('latitude')
            if latitude.bounds is None:
                latitude.guess_bounds()
            latitude.points = latitude.points.astype(np.float32).astype(
                np.float64)
            latitude.bounds = latitude.bounds.astype(np.float32).astype(
                np.float64)
            longitude = cube.coord('longitude')
            if longitude.bounds is None:
                longitude.guess_bounds()
            longitude.points = longitude.points.astype(np.float32).astype(
                np.float64)
            longitude.bounds = longitude.bounds.astype(np.float32).astype(
                np.float64)
        return cubes
