
"""Fixes for GFDL ESM2M."""

from ..fix import Fix
from ..cmip5.gfdl_esm2g import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all variables."""


class Sftof(Fix):
    """Fixes for sftof."""

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
        cube *= 1e6
        cube.metadata = metadata
        return cube


class Tos(Fix):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong standard_name.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        self.get_cube_from_list(cubes).standard_name = 'sea_surface_temperature'
        return cubes
