"""Fixes for CanESM5 model."""
import numpy as np
from ..fix import Fix


class Co2(Fix):
    """Fixes for co2."""

    def fix_data(self, cube):
        """Convert units from ppmv to 1.

        Parameters
        ----------
        cube : iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1.e-6
        cube.metadata = metadata
        return cubes


class Gpp(Fix):
    """Fixes for gpp, ocean values set to 0 instead of masked."""

    def fix_data(self, cube):
        """Fix masked values.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = np.ma.masked_where(cube.data == 0, cube.data)
        return cube
