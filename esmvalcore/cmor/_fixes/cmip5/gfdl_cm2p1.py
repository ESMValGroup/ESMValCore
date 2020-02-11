"""Fixes for GFDL CM2p1 model."""
from copy import deepcopy
import numpy as np
import cftime

from ..fix import Fix
from ..cmip5.gfdl_esm2g import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all variables."""


class Areacello(Fix):
    """Fixes for areacello"""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong units.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.units = 'm2'
        return cubes


class Sftof(Fix):
    """Fixes for sftof."""

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


class Sit(Fix):
    """Fixes for sit"""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes bad bounds

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        time = cube.coord('time')
        if self._fix_required(time):
            new_bounds = np.empty(time.bounds.shape, time.bounds.dtype)
            for x, point in enumerate(time):
                date = point.units.num2date(point.points[0])
                start = cftime.DatetimeJulian(date.year, date.month, 1)
                if date.month < 12:
                    end = cftime.DatetimeJulian(date.year, date.month + 1, 1)
                else:
                    end = cftime.DatetimeJulian(date.year + 1, 1, 1)
                new_bounds[x, 0] = point.units.date2num(start)
                new_bounds[x, 1] = point.units.date2num(end)
            time.bounds = new_bounds
        return cubes

    def _fix_required(self, time):
        if self.vardef.frequency != 'mon':
            return False
        if time.bounds[-1, 0] > time.points[-1]:
            return True
        if time.bounds[-1, 1] < time.points[-1]:
            return True
        return False


class Tos(Fix):
    """Fixes for tos"""

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
        metadata = deepcopy(cube.metadata)
        cube += 273.15
        cube.metadata = metadata
        return cube

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong standard_name.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.standard_name = 'sea_surface_temperature'
        return cubes
