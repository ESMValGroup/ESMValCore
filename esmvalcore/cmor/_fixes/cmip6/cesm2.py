"""Fixes for CESM2 model."""
from shutil import copyfile

import numpy as np
from netCDF4 import Dataset

from ..common import SiconcFixScalarCoord
from ..fix import Fix
from ..shared import (
    add_scalar_depth_coord,
    add_scalar_height_coord,
    add_scalar_typeland_coord,
    add_scalar_typesea_coord,
    fix_ocean_depth_coord,
)


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

    def fix_metadata(self, cubes):
        """Fix ``atmosphere_hybrid_sigma_pressure_coordinate``.

        See discussion in #882 for more details on that.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        lev_coord = cube.coord(var_name='lev')
        a_coord = cube.coord(var_name='a')
        b_coord = cube.coord(var_name='b')
        lev_coord.points = a_coord.core_points() + b_coord.core_points()
        lev_coord.bounds = a_coord.core_bounds() + b_coord.core_bounds()
        lev_coord.units = '1'
        return cubes


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


class Prw(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            latitude = cube.coord('latitude')
            if latitude.bounds is None:
                latitude.guess_bounds()
            latitude.bounds = latitude.bounds.astype(np.float64)
            latitude.bounds = np.round(latitude.bounds, 4)
            longitude = cube.coord('longitude')
            if longitude.bounds is None:
                longitude.guess_bounds()
            longitude.bounds = longitude.bounds.astype(np.float64)
            longitude.bounds = np.round(longitude.bounds, 4)

        return cubes


class Tas(Prw):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate.

        Fix also done for prw.
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        super().fix_metadata(cubes)
        # Specific code for tas
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


Siconc = SiconcFixScalarCoord


class Tos(Fix):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """
        Round times to 1 d.p. for monthly means.

        Required to get hist-GHG and ssp245-GHG Omon tos to concatenate.

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
            if cube.attributes['mipTable'] == 'Omon':
                cube.coord('time').points = \
                    np.round(cube.coord('time').points, 1)
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

                # Only points need to be fixed, not bounds
                if z_coord.units == 'cm':
                    z_coord.points = z_coord.core_points() / 100.0
                    z_coord.units = 'm'

                # Fix depth metadata
                if z_coord.standard_name is None:
                    fix_ocean_depth_coord(cube)
        return cubes
