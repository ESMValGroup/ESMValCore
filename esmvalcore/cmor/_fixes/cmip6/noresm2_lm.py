"""Fixes for NorESM2-LM model."""
import numpy as np
from ..fix import Fix


class Siconc(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Some coordinate points vary for different files of this dataset (for
        different time range). This fix removes these inaccuracies by rounding
        the coordinates.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes to fix.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            latitude = cube.coord('latitude')
            latitude.bounds = np.round(latitude.bounds, 4)
            longitude = cube.coord('longitude')
            longitude.bounds = np.round(longitude.bounds, 4)

        return cubes
