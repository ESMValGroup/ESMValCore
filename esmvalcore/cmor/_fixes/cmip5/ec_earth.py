"""Fixes for EC-Earth model."""
from dask import array as da
import iris

from ..fix import Fix
from ..shared import add_scalar_height_coord, cube_to_aux_coord


class Sic(Fix):
    """Fixes for sic."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """
        Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 273.15)
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """

        for cube in cubes:
            if not cube.coords(var_name='height'):
                add_scalar_height_coord(cube)

            if cube.coord('time').long_name is None:
                cube.coord('time').long_name = 'time'

        return cubes


class Areacello(Fix):
    """Fixes for areacello."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        areacello = cubes.extract('Areas of grid cell')[0]
        lat = cubes.extract('latitude')[0]
        lon = cubes.extract('longitude')[0]

        areacello.add_aux_coord(cube_to_aux_coord(lat), (0, 1))
        areacello.add_aux_coord(cube_to_aux_coord(lon), (0, 1))

        return iris.cube.CubeList([areacello, ])
