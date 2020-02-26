"""Fixes for MPI-ESM1-2-HR model."""

from ..fix import Fix
from ..shared import add_scalar_height_coord


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Adds missing height2m coordinate.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube)

        return cubes


class Ta(Fix):
    """Fixes for ta."""

    def fix_metadata(self, cubes):
        """
        Corrects plev coordinate var_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            plev = cube.coord('air_pressure')
            plev.var_name = 'plev'

        return cubes


class Va(Ta):
    """Fixes for va."""


class Zg(Ta):
    """Fixes for zg."""


class Ua(Ta):
    """Fixes for ua."""


class SfcWind(Fix):
    """Fixes for sfcWind."""

    def fix_metadata(self, cubes):
        """
        Adds missing height10m coordinate.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube, height=10.0)

        return cubes
