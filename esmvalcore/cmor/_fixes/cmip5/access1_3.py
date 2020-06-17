"""Fixes for ACCESS1-3 model."""
import iris
from cf_units import Unit

from ..fix import Fix
from .access1_0 import Cl as BaseCl


Cl = BaseCl


class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fixes wrong calendar 'gregorian' instead of 'proleptic_gregorian'

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            try:
                time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            else:
                if time.units.calendar == 'proleptic_gregorian':
                    time.convert_units(Unit("days since 1850-01-01",
                                            calendar='proleptic_gregorian'))
                    time.units = Unit(time.units.name, 'gregorian')
        return cubes
