"""Fixes for FGOALS-f3-L model."""
from ..fix import Fix
# from esmvalcore.cmor.fix import Fix

class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_data(self, cube):
        """Fix data.

        Unit is %, values are <= 1.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix

        Returns
        -------
        iris.cube.Cube
            Fixed cube. It can be a difference instance.
        """
        if cube.units == "%" and cube.data.max() <= 1.:
            cube.data = cube.data * 100
        return cube
