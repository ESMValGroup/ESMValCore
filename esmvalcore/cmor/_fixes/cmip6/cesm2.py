"""Fixes for CESM2 model."""
from shutil import copyfile

from netCDF4 import Dataset

from ..fix import Fix
from ..shared import (add_scalar_depth_coord, add_scalar_height_coord,
                      add_scalar_typeland_coord, add_scalar_typesea_coord)
from esmvalcore.preprocessor._shared import guess_bounds


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
        for cube in cubes:
            cube.coord('latitude').bounds=None
            cube.coord('longitude').bounds=None
            cube = guess_bounds(cube, ['latitude', 'longitude'])
        return cubes


class Pr(Fix):
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
        for cube in cubes:
            cube.coord('latitude').bounds=None
            cube.coord('longitude').bounds=None
            cube = guess_bounds(cube, ['latitude', 'longitude'])
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
