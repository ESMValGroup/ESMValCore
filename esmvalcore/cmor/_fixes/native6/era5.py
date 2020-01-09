"""Fixes for ERA5."""
import numpy as np
from iris.cube import CubeList

from ..fix import Fix
from ..shared import add_scalar_height_coord


class FixEra5(Fix):
    """Fixes for ERA5 variables"""

    @staticmethod
    def _frequency(cube):
        if not cube.coords(axis='T'):
            return 'fx'
        coord = cube.coord(axis='T')
        if 27 < coord.points[1] - coord.points[0] < 32:
            return 'monthly'
        return 'hourly'

class Accumulated(FixEra5):
    """Fixes for accumulated variables."""

    def _fix_frequency(self, cube):
        if self._frequency(cube) == 'monthly':
            cube.units = cube.units * 'd-1'
        elif self._frequency(cube) == 'hourly':
            cube.units = cube.units * 'h-1'
        return cube

    def fix_metadata(self, cubes):
        super().fix_metadata(cubes)
        for cube in cubes:
            self._fix_frequency(cube)
        return cubes

class Hydrological(FixEra5):
    """Fixes for hydrological variables."""

    @staticmethod
    def _fix_units(cube):
        cube.units = 'kg m-2'
        cube.data = cube.core_data() * 1000.
        return cube

    def fix_metadata(self, cubes):
        super().fix_metadata(cubes)
        for cube in cubes:
            self._fix_units(cube)
        return cubes

class Radiation(FixEra5):
    """Fixes for accumulated radiation variables."""

    def fix_metadata(self, cubes):
        super().fix_metadata(cubes)
        for cube in cubes:
            cube.attributes['positive'] = 'down'
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

        # Fix coordinate units, dtypes and names
        for coord_def in self.vardef.coordinates.values():
            coord_name = coord_def.standard_name
            coord = cube.coord(coord_name)
            if coord_def.axis == 'T':
                coord.convert_units('days since 1850-1-1 00:00:00.0')
            if coord_def.axis == 'Z':
                coord.convert_units(coord_def.units)
            coord.standard_name = coord_def.standard_name
            coord.var_name = coord_def.out_name
            coord.long_name = coord_def.long_name
            coord.points = coord.core_points().astype('float64')
            if len(coord.points) > 1:
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
        fixed_cubes = CubeList()
        for cube in cubes:
            cube.var_name = self.vardef.short_name
            cube.standard_name = self.vardef.standard_name
            cube.long_name = self.vardef.long_name

            cube = self._fix_coordinates(cube)
            self._fix_units(cube)

            cube.data = cube.core_data().astype('float32')

            fixed_cubes.append(cube)

        return fixed_cubes
