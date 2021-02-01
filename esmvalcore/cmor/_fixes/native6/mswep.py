"""Fixes for MSWEP."""
from datetime import datetime

import cf_units
import numpy as np
from cf_units import Unit

from ..fix import Fix


def fix_time_month(cube):
    """Fix time coordinates for monthly values.

    Convert from months since 1899-12 to days since 1850 as per CMOR
    standard.
    """
    time_coord = cube.coord('time')
    origin = time_coord.units.origin

    origin_year, origin_month = [
        int(val) for val in origin.split()[2].split('-')
    ]

    dates = []

    for time_point in time_coord.points:
        new_year = origin_year + (origin_month - 1 + time_point) // 12
        new_month = (origin_month - 1 + time_point) % 12 + 1
        dates.append(datetime(int(new_year), int(new_month), 15))

    t_unit = cf_units.Unit("days since 1850-01-01", calendar="standard")

    cube.coord('time').points = t_unit.date2num(dates)
    cube.coord('time').units = t_unit


def fix_time_day(cube):
    """Fix time coordinates for monthly values.

    Convert from days since 1899-12-31 to days since 1850 as per CMOR
    standard.
    """
    time_coord = cube.coord('time')
    time_coord.convert_units('days since 1850-1-1 00:00:00.0')


class Pr(Fix):
    """Fixes for pr."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        self._init_frequency_specific_fixes()

        for cube in cubes:
            self._fix_names(cube)
            self._fix_units(cube)
            self._fix_time(cube)
            self._fix_longitude(cube)
            self._fix_bounds(cube)

        return cubes

    def _init_frequency_specific_fixes(self):
        """Set frequency specific fixes (day/mon)."""
        frequency = self.vardef.frequency

        if frequency in ('day', '3hr'):
            self._fix_units = self._fix_units_day
            self._fix_time = fix_time_day
        elif frequency == 'mon':
            self._fix_units = self._fix_units_month
            self._fix_time = fix_time_month
        else:
            raise ValueError(f'No fixes for frequency: {frequency!r}')

    def _fix_units_month(self, cube):
        """Convert units from mm/month to kg m-2 s-1 units."""
        cube.units = Unit(self.vardef.units)
        # divide by number of seconds in a month
        cube.data = cube.core_data() / 60 * 60 * 24 * 30

    def _fix_units_day(self, cube):
        """Convert units from mm/month to kg m-2 s-1 units."""
        cube.units = Unit(self.vardef.units)
        # divide by number of seconds in a day
        cube.data = cube.core_data() / 60 * 60 * 24

    def _fix_longitude(self, cube):
        """Fix longitude coordinate from -180:180 to 0:360."""
        lon_axis = cube.coord_dims('longitude')
        lon = cube.coord(axis='X')

        assert lon.is_monotonic(), "Data must be monotonic to fix longitude."

        # roll data because iris forces `lon.points` to be strictly monotonic.
        shift = np.sum(lon.points < 0)
        points = np.roll(lon.points, -shift) % 360
        cube.data = np.roll(cube.core_data(), -shift, axis=lon_axis)

        lon.points = points

    def _fix_bounds(self, cube):
        """Add bounds to coords."""
        coord_defs = tuple(coord_def
                           for coord_def in self.vardef.coordinates.values())

        for coord_def in coord_defs:
            if not coord_def.must_have_bounds == 'yes':
                continue

            coord = cube.coord(axis=coord_def.axis)

            if coord.bounds is None:
                coord.guess_bounds()

    def _fix_names(self, cube):
        """Fix miscellaneous."""
        cube.var_name = self.vardef.short_name
        cube.standard_name = self.vardef.standard_name
        cube.long_name = self.vardef.long_name

        coord_defs = tuple(coord_def
                           for coord_def in self.vardef.coordinates.values())

        for coord_def in coord_defs:
            coord = cube.coord(axis=coord_def.axis)
            if not coord.long_name:
                coord.long_name = coord_def.long_name
