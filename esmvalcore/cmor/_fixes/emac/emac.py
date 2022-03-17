"""On-the-fly CMORizer for EMAC."""

import logging

import dask.array as da
import iris
import iris.cube
import iris.util

from ..fix import Fix
from ..shared import add_scalar_height_coord, add_scalar_typesi_coord

logger = logging.getLogger(__name__)


class EmacFix(Fix):
    """Base class for all EMAC fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
            if not cubes.extract(iris.NameConstraint(var_name=var_name)):
                raise ValueError(
                    f"Variable '{var_name}' used to extract "
                    f"'{self.vardef.short_name}' is not available in input "
                    f"file")
        return cubes.extract_cube(iris.NameConstraint(var_name=var_name))


class AllVars(EmacFix):
    """Fixes for all variables."""

    def fix_data(self, cube):
        """Fix data."""
        # Fix mask by masking all values where the absolute value is greater
        # than a given threshold (affects mostly 3D variables)
        mask_threshold = 1e20
        cube.data = da.ma.masked_outside(
            cube.core_data(), -mask_threshold, mask_threshold,
        )
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
            self._fix_time(cube)

        # Fix pressure levels (considers plev19, plev39, etc.)
        for dim_name in self.vardef.dimensions:
            if 'plev' in dim_name:
                self._fix_plev(cube)
                break

        # Fix latitude
        if 'latitude' in self.vardef.dimensions:
            lat_name = self.extra_facets.get('latitude', 'latitude')
            self._fix_lat(cube, lat_name)

        # Fix longitude
        if 'longitude' in self.vardef.dimensions:
            lon_name = self.extra_facets.get('longitude', 'longitude')
            self._fix_lon(cube, lon_name)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return iris.cube.CubeList([cube])

    @staticmethod
    def _fix_lat(cube, lat_name):
        """Fix latitude coordinate of cube."""
        lat = cube.coord(lat_name)
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
    def _fix_lon(cube, lon_name):
        """Fix longitude coordinate of cube."""
        lon = cube.coord(lon_name)
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

    def _fix_plev(self, cube):
        """Fix pressure level coordinate of cube."""
        for coord in cube.coords():
            coord_type = iris.util.guess_coord_axis(coord)
            if coord_type != 'Z':
                continue
            if not coord.units.is_convertible('Pa'):
                continue
            coord.var_name = 'plev'
            coord.standard_name = 'air_pressure'
            coord.lon_name = 'pressure'
            coord.convert_units('Pa')
            return
        raise ValueError(
            f"Cannot find requested pressure level coordinate for variable "
            f"'{self.vardef.short_name}', searched for Z coordinates with "
            f"units that are convertible to Pa")

    def _fix_scalar_coords(self, cube):
        """Fix scalar coordinates."""
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if 'typesi' in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

    @staticmethod
    def _fix_time(cube):
        """Fix time coordinate of cube."""
        time_coord = cube.coord('time')
        time_coord.var_name = 'time'
        time_coord.standard_name = 'time'
        time_coord.long_name = 'time'

        # Add bounds if possible (not possible if cube only contains single
        # time point)
        if not time_coord.has_bounds():
            try:
                time_coord.guess_bounds()
            except ValueError:
                pass

    def _fix_var_metadata(self, cube):
        """Fix metadata of variable."""
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name
        if cube.units != self.vardef.units:
            cube.convert_units(self.vardef.units)


class Siconc(EmacFix):
    """Fixes for ``siconc``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Note: This fix is called before the AllVars() fix. The wrong var_name
        # and units (which need to be %) are fixed in a later step in
        # AllVars(). This fix here is necessary to fix the "unknown" units that
        # cannot be converted to % in AllVars().
        cube = self.get_cube(cubes)
        cube.units = '1'
        return cubes


Siconca = Siconc
