"""Fixes for GFDL CM2p1 model."""
from copy import deepcopy

import cftime
import numpy as np

from esmvalcore.iris_helpers import date2num

from ..cmip5.gfdl_esm2g import AllVars as BaseAllVars
from ..fix import Fix
from .cesm1_cam5 import Cl as BaseCl

Cl = BaseCl


class AllVars(BaseAllVars):
    """Fixes for all variables."""


class Areacello(Fix):
    """Fixes for areacello."""

    def fix_metadata(self, cubes):
        """Fix metadata.

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
        """Fix data.

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
    """Fixes for sit."""

    def fix_metadata(self, cubes):
        """Fix metadata.

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
            times = time.units.num2date(time.points)
            starts = [
                cftime.DatetimeJulian(c.year, c.month, 1)
                for c in times
            ]
            ends = [
                cftime.DatetimeJulian(c.year, c.month + 1, 1)
                if c.month < 12
                else cftime.DatetimeJulian(c.year + 1, 1, 1)
                for c in times
            ]
            time.bounds = date2num(np.stack([starts, ends], -1), time.units)
        return cubes

    def _fix_required(self, time):
        return (
            self.vardef.frequency == 'mon' and
            not (time.bounds[-1, 0] < time.points[-1] < time.bounds[-1, 1])
        )


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """Fix data.

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
        """Fix metadata.

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
