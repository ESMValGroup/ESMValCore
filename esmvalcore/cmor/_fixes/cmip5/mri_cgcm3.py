"""Fixes for MRI-CGCM3 model."""
from dask import array as da

from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


class Msftmyz(Fix):
    """Fixes for msftmyz."""

    def fix_data(self, cube):
        """
        Fix msftmyz data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.)
        return cube


class ThetaO(Fix):
    """Fixes for thetao."""

    def fix_data(self, cube):
        """
        Fix thetao data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.)
        return cube
