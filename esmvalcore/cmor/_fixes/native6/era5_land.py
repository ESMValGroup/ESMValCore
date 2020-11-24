"""Fixes for ERA5-Land."""
import datetime
import logging

import iris
import numpy as np

from ..fix import Fix
from ..shared import add_scalar_height_coord
from era5 import (get_frequency,
                  fix_hourly_time_coordinate,
                  fix_accumulated_units,
                  multiply_with_density)

logger = logging.getLogger(__name__)


class Pr(Fix):
    """Fixes for pr."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            fix_hourly_time_coordinate(cube)
            fix_accumulated_units(cube)
            multiply_with_density(cube)

        return cubes


class AllVars(Fix):
    """Fixes for all variables."""
    def _fix_coordinates(self, cube):
        """Fix coordinates."""
        # Fix coordinate increasing direction
        slices = []
        for coord in cube.coords():
            if coord.var_name in ('latitude', 'pressure_level'):
                slices.append(slice(None, None, -1))
            else:
                slices.append(slice(None))
        cube = cube[tuple(slices)]

        # Add scalar height coordinates
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.)

        for coord_def in self.vardef.coordinates.values():
            axis = coord_def.axis
            coord = cube.coord(axis=axis)
            if axis == 'T':
                coord.convert_units('days since 1850-1-1 00:00:00.0')
            if axis == 'Z':
                coord.convert_units(coord_def.units)
            coord.standard_name = coord_def.standard_name
            coord.var_name = coord_def.out_name
            coord.long_name = coord_def.long_name
            coord.points = coord.core_points().astype('float64')
            if (coord.bounds is None and len(coord.points) > 1
                    and coord_def.must_have_bounds == "yes"):
                coord.guess_bounds()

        self._fix_monthly_time_coord(cube)

        return cube

    @staticmethod
    def _fix_monthly_time_coord(cube):
        """Set the monthly time coordinates to the middle of the month."""
        if get_frequency(cube) == 'monthly':
            coord = cube.coord(axis='T')
            end = []
            for cell in coord.cells():
                month = cell.point.month + 1
                year = cell.point.year
                if month == 13:
                    month = 1
                    year = year + 1
                end.append(cell.point.replace(month=month, year=year))
            end = coord.units.date2num(end)
            start = coord.points
            coord.points = 0.5 * (start + end)
            coord.bounds = np.column_stack([start, end])

    def _fix_units(self, cube):
        """Fix units."""
        cube.convert_units(self.vardef.units)

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()
        for cube in cubes:
            cube.var_name = self.vardef.short_name
            if self.vardef.standard_name:
                cube.standard_name = self.vardef.standard_name
            cube.long_name = self.vardef.long_name

            cube = self._fix_coordinates(cube)
            self._fix_units(cube)

            cube.data = cube.core_data().astype('float32')
            year = datetime.datetime.now().year
            cube.attributes['comment'] = (
                'Contains modified Copernicus Climate Change '
                f'Service Information {year}')

            fixed_cubes.append(cube)

        return fixed_cubes
