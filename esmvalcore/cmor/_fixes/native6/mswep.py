"""Fixes for MSWEP."""
from datetime import datetime

import cf_units
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
        frequency = self.vardef.frequency

        if frequency == 'day':
            fix_time = fix_time_day
            fix_units = self._fix_units_day
        elif frequency == 'mon':
            fix_units = self._fix_units_month
            fix_time = fix_time_month
        else:
            raise ValueError(f'Cannot fix frequency: {frequency!r}')

        for cube in cubes:
            cube.var_name = self.vardef.short_name
            cube.standard_name = self.vardef.standard_name
            cube.long_name = self.vardef.long_name

            fix_units(cube)
            fix_time(cube)

        return cubes

    def _fix_units_month(self, cube):
        """Convert units from mm/month to kg m-3 s-1 units."""
        cube.units = Unit(self.vardef.units)
        # divide by number of seconds in a month
        cube.data = cube.core_data() / 60 * 60 * 24 * 30

    def _fix_units_day(self, cube):
        """Convert units from mm/month to kg m-3 s-1 units."""
        cube.units = Unit(self.vardef.units)
        # divide by number of seconds in a day
        cube.data = cube.core_data() / 60 * 60 * 24
