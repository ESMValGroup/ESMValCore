"""Fixes for CESM2 model."""
from shutil import copyfile

from netCDF4 import Dataset

from ..fix import Fix
from ..shared import (add_scalar_depth_coord, add_scalar_height_coord,
                      add_scalar_typeland_coord, add_scalar_typesea_coord)
import cf_units
import numpy as np
import iris

class Cl(Fix):
    """Fixes for ``cl``."""

    def _fix_formula_terms(self, filepath, output_dir):
        """Fix ``formula_terms`` attribute."""
        new_path = self.get_fixed_filepath(output_dir, filepath)
        copyfile(filepath, new_path)
        dataset = Dataset(new_path, mode='a')
        dataset.variables['lev'].formula_terms = 'p0: p0 a: a b: b ps: ps'
        dataset.variables['lev'].standard_name = (
            'atmosphere_hybrid_sigma_pressure_coordinate')
        dataset.close()
        return new_path

    def fix_data(self, cube):
        """Fix data.

        Fixed ordering of vertical coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        (z_axis,) = cube.coord_dims(cube.coord(axis='Z', dim_coords=True))
        indices = [slice(None)] * cube.ndim
        indices[z_axis] = slice(None, None, -1)
        cube = cube[tuple(indices)]
        return cube

    def fix_file(self, filepath, output_dir):
        """Fix hybrid pressure coordinate.

        Adds missing ``formula_terms`` attribute to file.

        Note
        ----
        Fixing this with :mod:`iris` in ``fix_metadata`` or ``fix_data`` is
        **not** possible, since the bounds of the vertical coordinates ``a``
        and ``b`` are not present in the loaded :class:`iris.cube.CubeList`,
        even when :func:`iris.load_raw` is used.

        Parameters
        ----------
        filepath : str
            Path to the original file.
        output_dir : str
            Path of the directory where the fixed file is saved to.

        Returns
        -------
        str
            Path to the fixed file.

        """
        new_path = self._fix_formula_terms(filepath, output_dir)
        dataset = Dataset(new_path, mode='a')
        dataset.variables['a_bnds'][:] = dataset.variables['a_bnds'][::-1, :]
        dataset.variables['b_bnds'][:] = dataset.variables['b_bnds'][::-1, :]
        dataset.close()
        return new_path


Cli = Cl


Clw = Cl


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
