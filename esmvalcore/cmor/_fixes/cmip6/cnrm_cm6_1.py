"""Fixes for CNRM-CM6-1 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import (
    add_aux_coords_from_cubes,
    get_bounds_cube,
    fix_ocean_depth_coord
)


class Cl(ClFixHybridPressureCoord):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix vertical hybrid sigma coordinate (incl. bounds).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)

        # Add auxiliary coordinate from list of cubes
        coords_to_add = {
            'ap': 1,
            'b': 1,
            'ps': (0, 2, 3),
        }
        add_aux_coords_from_cubes(cube, cubes, coords_to_add)
        cube.coord(var_name='ap').units = 'Pa'

        # Fix vertical coordinate bounds
        for coord_name in ('ap', 'b'):
            bounds_cube = get_bounds_cube(cubes, coord_name)
            bounds = bounds_cube.data.reshape(-1, 2)
            new_bounds_cube = iris.cube.Cube(bounds,
                                             **bounds_cube.metadata._asdict())
            cubes.remove(bounds_cube)
            cubes.append(new_bounds_cube)

        # Fix hybrid sigma pressure coordinate
        cubes = super().fix_metadata(cubes)

        # Fix horizontal coordinates bounds
        for coord_name in ('latitude', 'longitude'):
            cube.coord(coord_name).guess_bounds()
        return cubes


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
        alt_40_coord = cube.coord('alt40')
        alt_40_coord.standard_name = 'altitude'
        return iris.cube.CubeList([cube])


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


Cli = Cl

Clw = Cl
