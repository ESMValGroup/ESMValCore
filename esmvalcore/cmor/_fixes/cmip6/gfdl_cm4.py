"""Fixes for GFDL-CM4 model."""
import iris

from ..common import ClFixHybridPressureCoord, SiconcFixScalarCoord
from ..fix import Fix
from ..shared import add_aux_coords_from_cubes, add_scalar_height_coord
from ..shared import fix_ocean_depth_coord

class Cl(ClFixHybridPressureCoord):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix hybrid sigma pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        coords_to_add = {
            'ap': 1,
            'b': 1,
            'ps': (0, 2, 3),
        }
        add_aux_coords_from_cubes(cube, cubes, coords_to_add)
        return super().fix_metadata(cubes)


Cli = Cl


Clw = Cl


Siconc = SiconcFixScalarCoord


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        try:
            cube.coord('height')
        except iris.exceptions.CoordinateNotFoundError:
            add_scalar_height_coord(cube, 2.0)
        return cubes


class Uas(Fix):
    """Fixes for uas."""

    def fix_metadata(self, cubes):
        """
        Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


class Vas(Fix):
    """Fixes for vas."""

    def fix_metadata(self, cubes):
        """
        Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
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
            if cube.coords('latitude'):
                cube.coord('latitude').var_name = 'lat'
            if cube.coords('longitude'):
                cube.coord('longitude').var_name = 'lon'

            if cube.coords(axis='Z'):
#                z_coord = cube.coord(axis='Z')
#                if z_coord.var_name == 'olevel':
                fix_ocean_depth_coord(cube)
        return cubes


