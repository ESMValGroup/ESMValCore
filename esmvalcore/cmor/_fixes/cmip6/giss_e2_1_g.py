"""Fixes for GISS-E2-1-G model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord

class Nbp(Fix):
    """Fixes for nbp."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes wrong sign for land surface flux.

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
