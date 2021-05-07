"""Fixes for IPSL-CM6A-LR model."""
from iris.cube import CubeList

from ..fix import Fix
from ..shared import fix_ocean_depth_coord


class AllVars(Fix):
    """Fixes for thetao."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.coords('latitude'):
            cube.coord('latitude').var_name = 'lat'
        if cube.coords('longitude'):
            cube.coord('longitude').var_name = 'lon'
        return CubeList([cube])


class Clcalipso(Fix):
    """Fixes for ``clcalipso``."""

    def fix_metadata(self, cubes):
        """Fix ``alt40`` coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        alt_40_coord = cube.coord('height')
        alt_40_coord.long_name = 'altitude'
        alt_40_coord.standard_name = 'altitude'
        alt_40_coord.var_name = 'alt40'
        return CubeList([cube])


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
                if z_coord.var_name == 'olevel':
                    fix_ocean_depth_coord(cube)
        return cubes
