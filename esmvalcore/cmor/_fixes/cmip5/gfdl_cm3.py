
"""Fixes for GFDL CM3 model."""
from ..fix import Fix

from ..cmip5.gfdl_esm2g import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all variables."""


class Sftof(Fix):
    """Fix sftof."""

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
