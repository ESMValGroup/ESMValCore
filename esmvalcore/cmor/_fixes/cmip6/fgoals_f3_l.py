"""Fixes for FGOALS-f3-L model."""
from ..fix import Fix
import dask.array as da
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
        if cube.units == "%" and da.max(cube.core_data()).compute() <= 1.:
            cube.data = cube.core_data() * 100.
        return cube
