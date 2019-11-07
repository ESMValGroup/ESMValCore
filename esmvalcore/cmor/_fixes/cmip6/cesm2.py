"""Fixes for CESM2 model."""
from ..fix import Fix
from ..shared import (add_scalar_depth_coord, add_scalar_height_coord,
                      add_scalar_typeland_coord, add_scalar_typesea_coord)
import cf_units
import numpy as np
import iris

class Fgco2(Fix):
    """Fixes for fgco2."""
    def fix_metadata(self, cubes):
        """Add depth (0m) coordinate.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

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
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

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
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

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
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typesea_coord(cube)
        return cubes


class msftmz(Fix):
    """Fix msftmz."""

    def fix_metadata(self, cubes):
        """
        Problems:
         basin has incorrect long name, var.
         Dimensions are also wrong.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        new_cubes = []
        for cube in cubes:

            # Fix regions coordinate
            cube.remove_coord(cube.coord("region"))
            values = np.array(['atlantic_arctic_ocean', 'indian_pacific_ocean',
                               'global_ocean',], dtype='<U21')
            basin_coord = iris.coords.AuxCoord(
                values,
                standard_name=u'region',
                units=cf_units.Unit('no_unit'),
                long_name=u'ocean basin',
                var_name='basin')

            # Replace broken coord with correct one.
            cube.add_aux_coord(basin_coord, data_dims=1)
            print(cube.ndim)
            print(cube)
            print(cube.coord("region"))

            # Fix depth
            depth = cube.coord('lev')
            depth.var_name = 'depth'
            depth.standard_name = 'depth'
            depth.long_name = 'depth'

        return cubes
