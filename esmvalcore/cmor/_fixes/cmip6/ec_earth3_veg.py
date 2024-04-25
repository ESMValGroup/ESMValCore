"""Fixes for EC-Earth3-Veg model."""
import cf_units
import numpy as np

from ..fix import Fix


class Siconca(Fix):
    """Fixes for siconca."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube which needs to be fixed.

        Returns
        -------
        iris.cube.Cube
        """
        cube.data = cube.core_data() * 100.
        return cube


class CalendarFix(Fix):
    """Use the same calendar for all files.

    The original files contain a mix of `gregorian` for the historical
    experiment and `proleptic_gregorian` for the ssp experiments.
    """

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            if cube.coords('time'):
                time_coord = cube.coord('time')
                time_coord.units = cf_units.Unit(time_coord.units.origin,
                                                 'proleptic_gregorian')
        return cubes


class Siconc(CalendarFix):
    """Fixes for siconc variable."""


class Tos(CalendarFix):
    """Fixes for tos."""


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix latitude points and bounds.

        Fix latitude_bounds and longitude_bounds data type and round to 8 d.p.

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
            latitude = cube.coord('latitude')
            latitude.points = np.round(latitude.core_points(), 8)
            latitude.bounds = np.round(latitude.core_bounds(), 8)

        return cubes
