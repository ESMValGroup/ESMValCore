
"""Fixes for MRI-ESM1 model."""
from dask import array as da

from ..fix import Fix


class Msftmyz(Fix):
    """Fixes for msftmyz."""

    def fix_data(self, cube):
        """
        Fix msftmyz data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.)
        return cube
