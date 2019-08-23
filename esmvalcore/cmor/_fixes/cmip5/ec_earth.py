"""Fixes for EC-Earth model."""
from dask import array as da

from ..fix import Fix


class Sic(Fix):
    """Fixes for sic."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """
        Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 273.15)
        return cube
