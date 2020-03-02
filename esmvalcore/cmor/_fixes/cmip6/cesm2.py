"""Fixes for CESM2 model."""
import iris

from ..cmip5.bcc_csm1_1 import Cl as BaseCl
from ..fix import Fix
from ..shared import (add_aux_coords_from_cubes, add_scalar_depth_coord,
                      add_scalar_height_coord, add_scalar_typeland_coord,
                      add_scalar_typesea_coord)


class Cl(BaseCl):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix vertical hybrid sigma coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)

        # Add auxiliary coordinate from list of cubes
        coords_to_add = {
            'a': 1,
            'b': 1,
            'ps': (0, 2, 3),
            'p0': (),
        }
        add_aux_coords_from_cubes(cube, cubes, coords_to_add)

        # Add ap coordinate
        a_coord = cube.coord(var_name='a')
        p0_coord = cube.coord(var_name='p0')
        ap_coord = a_coord * p0_coord.points[0]
        ap_coord.units = a_coord.units * p0_coord.units
        ap_coord.rename('vertical pressure')
        ap_coord.var_name = 'ap'
        cube.add_aux_coord(ap_coord, cube.coord_dims(a_coord))

        # Add aux_factory
        pressure_coord_factory = iris.aux_factory.HybridPressureFactory(
            delta=ap_coord,
            sigma=cube.coord(var_name='b'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        cube.add_aux_factory(pressure_coord_factory)
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


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube)
        return cubes


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_metadata(self, cubes):
        """Add typeland coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typeland_coord(cube)
        return cubes


class Sftof(Fix):
    """Fixes for sftof."""

    def fix_metadata(self, cubes):
        """Add typesea coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typesea_coord(cube)
        return cubes
