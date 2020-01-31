"""Fixes for ERA5."""
import datetime
import logging

import iris
import numpy as np

from ..fix import Fix
from ..shared import add_scalar_height_coord

logger = logging.getLogger(__name__)


class FixEra5(Fix):
    """Fixes for ERA5 variables."""
    @staticmethod
    def _frequency(cube):
        """Determine time frequency of input cube."""
        try:
            time = cube.coord(axis='T')
        except iris.exceptions.CoordinateNotFoundError:
            return 'fx'

        time.convert_units('days since 1850-1-1 00:00:00.0')
        if len(time.points) == 1:
            if cube.long_name != 'Geopotential':
                raise ValueError('Unable to infer frequency of cube '
                                 f'with length 1 time dimension: {cube}')
            return 'fx'
        interval = time.points[1] - time.points[0]
        if interval - 1 / 24 < 1e-4:
            return 'hourly'
        return 'monthly'


class Accumulated(FixEra5):
    """Fixes for accumulated variables."""
    def _fix_frequency(self, cube):
        if self._frequency(cube) == 'monthly':
            cube.units = cube.units * 'd-1'
        elif self._frequency(cube) == 'hourly':
            cube.units = cube.units * 'h-1'
        return cube

    def _fix_hourly_time_coordinate(self, cube):
        if self._frequency(cube) == 'hourly':
            time = cube.coord(axis='T')
            time.points = time.points - 1 / 48
            time.guess_bounds()
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        super().fix_metadata(cubes)
        for cube in cubes:
            self._fix_hourly_time_coordinate(cube)
            self._fix_frequency(cube)
        return cubes


class Hydrological(FixEra5):
    """Fixes for hydrological variables."""
    @staticmethod
    def _fix_units(cube):
        cube.units = 'kg m-2 s-1'
        cube.data = cube.core_data() * 1000.
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        super().fix_metadata(cubes)
        for cube in cubes:
            self._fix_units(cube)
        return cubes


class Radiation(FixEra5):
    """Fixes for accumulated radiation variables."""
    @staticmethod
    def _fix_direction(cube):
        cube.attributes['positive'] = 'down'

    def fix_metadata(self, cubes):
        """Fix metadata."""
        super().fix_metadata(cubes)
        for cube in cubes:
            self._fix_direction(cube)
        return cubes


class Fx(FixEra5):
    """Fixes for time invariant variables."""
    @staticmethod
    def _remove_time_coordinate(cube):
        cube = cube[0]
        cube.remove_coord('time')
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        squeezed_cubes = []
        for cube in cubes:
            cube = self._remove_time_coordinate(cube)
            squeezed_cubes.append(cube)
        return iris.cube.CubeList(squeezed_cubes)


class Tasmin(FixEra5):
    """Fixes for tasmin."""
    def fix_metadata(self, cubes):
        for cube in cubes:
            if self._frequency(cube) == 'hourly':
                time = cube.coord(axis='T')
                time.points = time.points - 1 / 48
                time.guess_bounds()
        return cubes


class Tasmax(FixEra5):
    """Fixes for tasmax."""
    def fix_metadata(self, cubes):
        for cube in cubes:
            if self._frequency(cube) == 'hourly':
                time = cube.coord(axis='T')
                time.points = time.points - 1 / 48
                time.guess_bounds()
        return cubes


class Evspsbl(Hydrological, Accumulated):
    """Fixes for evspsbl."""


class Mrro(Hydrological, Accumulated):
    """Fixes for evspsbl."""


class Prsn(Hydrological, Accumulated):
    """Fixes for evspsbl."""


class Pr(Hydrological, Accumulated):
    """Fixes for evspsbl."""


class Evspsblpot(Hydrological, Accumulated):
    """Fixes for evspsbl."""


class Rss(Radiation, Accumulated):
    """Fixes for Rss."""


class Rsds(Radiation, Accumulated):
    """Fixes for Rsds."""


class Rsdt(Radiation, Accumulated):
    """Fixes for Rsdt."""


class Rls(Radiation):
    """Fixes for Rls."""


class Orog(Fx):
    """Fixes for orography."""
    @staticmethod
    def _divide_by_gravity(cube):
        cube.units = cube.units / 'm s-2'
        cube.data = cube.core_data() / 9.80665
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cubes = super().fix_metadata(cubes)
        for cube in cubes:
            self._divide_by_gravity(cube)
        return cubes


class AllVars(FixEra5):
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

    def _fix_monthly_time_coord(self, cube):
        """Set the monthly time coordinates to the middle of the month."""
        if self._frequency(cube) == 'monthly':
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
