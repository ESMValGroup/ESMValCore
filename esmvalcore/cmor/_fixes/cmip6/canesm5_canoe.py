"""Fixes for CanESM5-CanOE model."""
from ..fix import Fix
import numpy as np


class Co2(Fix):
    """Fixes for co2."""
    def fix_metadata(self, cubes):
        """Corrects units.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        metadata = cube.metadata
        cube *= 1.e-6
        cube.metadata = metadata
        return cubes


class Gpp(Fix):
    """Fixes for gpp, land values set to 0 instead of masked."""

    def fix_data(self, cube):
        """
        Fix data.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = np.ma.masked_where(cube.data == 0, cube.data)
        return cube
