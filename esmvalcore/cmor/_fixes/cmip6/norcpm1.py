"""Fixes for CMIP6 NorCPM1."""
from ..fix import Fix
import numpy as np

class Fgco2(Fix):
    """Fixes for fgco2, land values set to 0 instead of masked."""

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
        cube.data = np.ma.masked_where(cube.data==0, cube.data)
        return cube
