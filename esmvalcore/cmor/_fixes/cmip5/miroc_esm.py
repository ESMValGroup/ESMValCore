"""Fixes for MIROC-ESM model."""

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError

from ..common import ClFixHybridPressureCoord
from ..fix import Fix

Cl = ClFixHybridPressureCoord


class Tro3(Fix):
    """Fixes for tro3."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1000
        cube.metadata = metadata
        return cube


class Co2(Fix):
    """Fixes for co2."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes error in cube units

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        self.get_cube_from_list(cubes).units = '1.0e-6'
        return cubes


class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes error in air_pressure coordinate, sometimes called AR5PL35, and
        error in time coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            # Fix air_pressure
            try:
                old = cube.coord('AR5PL35')
                dims = cube.coord_dims(old)
                cube.remove_coord(old)

                plev = DimCoord.from_coord(old)
                plev.var_name = 'plev'
                plev.standard_name = 'air_pressure'
                plev.long_name = 'pressure'
                cube.add_dim_coord(plev, dims)
            except CoordinateNotFoundError:
                pass

            # Fix time for files that contain year < 1 (which is not allowed)
            if cube.coords('time'):
                expected_time_units = Unit('days since 1950-1-1 00:00:00',
                                           calendar='gregorian')
                if cube.coord('time').units != expected_time_units:
                    continue
                if cube.coord('time').bounds is None:
                    continue

                # Only apply fix if there is a year < 1 in the first element
                # of the time bounds (-711860.5 days from 1950-01-01 is <
                # year 1)
                if np.isclose(cube.coord('time').bounds[0][0], -711860.5):
                    new_points = cube.coord('time').points.copy() + 3.5
                    new_bounds = cube.coord('time').bounds.copy() + 3.5
                    cube.coord('time').points = new_points
                    cube.coord('time').bounds = new_bounds

        return cubes
