# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for MRI-CGCM3 model."""
from dask import array as da

from ..fix import Fix


class msftmyz(Fix):
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


class thetao(Fix):
    """Fixes for thetao."""

    def fix_data(self, cube):
        """
        Fix thetao data.

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
