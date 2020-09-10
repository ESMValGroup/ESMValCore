"""Fixes for CMIP6 NorESM2-LM."""
from ..fix import Fix

class co2(Fix):
    """Fixes for co2."""

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
        metadata = cube.metadata
        cube *= 29./44.
        cube.metadata = metadata
        return cube
