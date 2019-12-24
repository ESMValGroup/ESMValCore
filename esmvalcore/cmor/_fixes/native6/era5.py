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

class Hydrological(Accumulated):
    """Fixes for accumulated hydrological variables."""

    def _fix_units(self, cube):
        cube.units = cube.units * 'kg m-3'
        cube.data = cube.core_data() * 1000.

    def fix_metadata(self, cubes):
        for cube in cubes:
            self._fix_frequency(cube)
            self._fix_units(cube)
        return cubes

# radiation fixes: 'rls', 'rsds', 'rsdt', 'rss'

class Radiation(Accumulated):
    """Fixes for accumulated radiation variables."""

    def fix_metadata(self, cubes):
        for cube in cubes:
            cube.attributes['positive'] = 'down'
            self._fix_frequency(cube)
        return cubes

class Evspsbl(Hydrological):
    """Fixes for evspsbl."""

class Mrro(Hydrological):
    """Fixes for evspsbl."""

class Prsn(Hydrological):
    """Fixes for evspsbl."""

class Pr(Hydrological):
    """Fixes for evspsbl."""

class Evspsblpot(Hydrological):
    """Fixes for evspsbl."""

class Rss(Radiation):
    """Fixes for Rss."""

class Rsds(Radiation):
    """Fixes for Rsds."""

class Rsdt(Radiation):
    """Fixes for Rsdt."""

class Rls(Radiation):
    """Fixes for Rls."""

    def fix_metadata(self, cubes):
        for cube in cubes:
            cube.attributes['positive'] = 'down'
        return cubes

class Clt(Fix):
    """Fixes for clt."""

    def fix_metadata(self, cubes):
        """Fix units."""
        for cube in cubes:
            cube.units = 1
        return cubes   

class AllVars(FixEra5):
    """Fixes for all variables."""

    def _fix_coordinates(self, cube):
        """Fix coordinates."""
        # Make latitude increasing
        cube = cube[..., ::-1, :]

        # Make pressure_levels decreasing
        if cube.coords('pressure_level'):
            cube = cube[:, ::-1, ...]

        # Add scalar height coordinates
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.)

        for axis in 'T', 'X', 'Y', 'Z':
            coord_def = self.vardef.coordinates.get(axis)
            if coord_def:
                coord = cube.coord(axis=axis)
                if axis == 'T':
                    coord.convert_units('days since 1850-1-1 00:00:00.0')
                if axis == 'Z':
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
        if cube.units == '(0 - 1)':
            # Correct dimensionless units to 1 from ecmwf format '(0 - 1)'
            cube.units = 1
        if cube.units == 'm of water equivalent':
            # Correct units from m of water to kg of water per m2
            cube.units = 'kg m-2'
            cube.data = cube.core_data() * 1000.

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
