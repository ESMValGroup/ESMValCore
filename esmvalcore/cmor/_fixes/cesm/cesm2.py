"""On-the-fly CMORizer for CESM2."""

import logging

from iris.cube import CubeList

from ..shared import (
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
    add_scalar_typesi_coord,
)
from ._base_fixes import CesmFix

logger = logging.getLogger(__name__)


INVALID_UNITS = {
    'fraction': '1',
}


class AllVars(CesmFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
            self._fix_time(cube)

        # Fix latitude
        if 'latitude' in self.vardef.dimensions:
            self._fix_lat(cube)

        # Fix longitude
        if 'longitude' in self.vardef.dimensions:
            self._fix_lon(cube)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return CubeList([cube])

    @staticmethod
    def _fix_time(cube):
        """Fix time coordinate of cube."""
        time_coord = cube.coord('time')
        time_coord.var_name = 'time'
        time_coord.standard_name = 'time'
        time_coord.long_name = 'time'

        # Move time points to center of time period given by time bounds
        # (currently the points are located at the end of the interval)
        if time_coord.bounds is not None:
            time_coord.points = time_coord.bounds.mean(axis=-1)

        # Add bounds if possible (not possible if cube only contains single
        # time point)
        if not time_coord.has_bounds():
            try:
                time_coord.guess_bounds()
            except ValueError:
                pass

    @staticmethod
    def _fix_lat(cube):
        """Fix latitude coordinate of cube."""
        lat = cube.coord('latitude')
        lat.var_name = 'lat'
        lat.standard_name = 'latitude'
        lat.long_name = 'latitude'
        lat.convert_units('degrees_north')

        # Add bounds if possible (not possible if cube only contains single
        # lat point)
        if not lat.has_bounds():
            try:
                lat.guess_bounds()
            except ValueError:
                pass

    @staticmethod
    def _fix_lon(cube):
        """Fix longitude coordinate of cube."""
        lon = cube.coord('longitude')
        lon.var_name = 'lon'
        lon.standard_name = 'longitude'
        lon.long_name = 'longitude'
        lon.convert_units('degrees_east')

        # Add bounds if possible (not possible if cube only contains single
        # lon point)
        if not lon.has_bounds():
            try:
                lon.guess_bounds()
            except ValueError:
                pass

    def _fix_scalar_coords(self, cube):
        """Fix scalar coordinates."""
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if 'lambda550nm' in self.vardef.dimensions:
            add_scalar_lambda550nm_coord(cube)
        if 'typesi' in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

    def _fix_var_metadata(self, cube):
        """Fix metadata of variable."""
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name

        # Fix units
        if 'invalid_units' in cube.attributes:
            invalid_units = cube.attributes.pop('invalid_units')
            new_units = INVALID_UNITS.get(invalid_units, invalid_units)
            try:
                cube.units = new_units
            except ValueError as exc:
                raise ValueError(
                    f"Failed to fix invalid units '{invalid_units}' for "
                    f"variable '{self.vardef.short_name}'") from exc
        if cube.units != self.vardef.units:
            cube.convert_units(self.vardef.units)

        # Fix attributes
        if self.vardef.positive != '':
            cube.attributes['positive'] = self.vardef.positive
