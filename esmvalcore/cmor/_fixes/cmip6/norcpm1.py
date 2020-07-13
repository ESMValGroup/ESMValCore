"""Fixes for NorCPM1."""
from ..fix import Fix

class Nbp(Fix):
    """Fixes for nbp."""

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
