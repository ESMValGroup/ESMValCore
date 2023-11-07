"""Fixes for MPI-ESM1-2-HR model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord, round_coordinates


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            if cube.attributes.get('variant_label', '') == 'r2i1p1f1':
                round_coordinates(
                    [cube],
                    decimals=11,
                    coord_names=['latitude'],
                )
        return cubes


Cl = ClFixHybridPressureCoord

Cli = ClFixHybridPressureCoord

Clw = ClFixHybridPressureCoord


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add missing height2m coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

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
        """Corrects plev coordinate var_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

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
        """Add missing height10m coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            add_scalar_height_coord(cube, height=10.0)

        return cubes
