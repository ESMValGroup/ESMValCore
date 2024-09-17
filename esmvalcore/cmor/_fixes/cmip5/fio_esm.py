
"""Fixes for FIO ESM model."""
from ..fix import Fix
from .cesm1_cam5 import Cl as BaseCl


Cl = BaseCl


class Co2(Fix):
    """Fixes for co2."""

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
        cube *= 29. / 44. * 1.e6
        cube.metadata = metadata
        return cube


class Ch4(Fix):
    """Fixes for ch4."""

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
        cube *= 29. / 16. * 1.e9
        cube.metadata = metadata
        return cube
