"""Fixes for GFDL-ESM4 model."""
import iris
from ..fix import Fix
from ..shared import (add_scalar_depth_coord, fix_ocean_depth_coord)


class Siconc(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """
        Fix missing type.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        typesi = iris.coords.AuxCoord(
            'siconc',
            standard_name='area_type',
            long_name='Sea Ice area type',
            var_name='type',
            units='1',
            bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typesi)
        return cubes


class Fgco2(Fix):
    """Fixes for fgco2."""

    def fix_metadata(self, cubes):
        """Add depth (0m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_depth_coord(cube)
        return cubes


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if z_coord.standard_name is None:
                    fix_ocean_depth_coord(cube)
        return cubes
