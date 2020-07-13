"""Fixes for GISS-E2-1-G-CC model."""
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
        cube *= -1
        cube.metadata = metadata
        return cube
