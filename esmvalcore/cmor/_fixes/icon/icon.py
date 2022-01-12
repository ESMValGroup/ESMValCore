"""On-the-fly CMORizer for ICON."""

import logging
from datetime import datetime

import cf_units
import dask.array as da
import iris
import numpy as np

from esmvalcore.iris_helpers import date2num, var_name_constraint

from ..fix import Fix
from ..shared import add_scalar_height_coord, add_scalar_typesi_coord

logger = logging.getLogger(__name__)


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        raw_name = self.extra_facets.get('raw_name', self.vardef.short_name)
        cube = cubes.extract_cube(var_name_constraint(raw_name))

        # Fix dimensional coordinates
        if cube.coords("time"):
            self._fix_time(cube)
        if cube.coords("height"):
            plev_points_cube = cubes.extract_cube(var_name_constraint('pfull'))
            plev_bounds_cube = cubes.extract_cube(var_name_constraint('phalf'))
            cube = self._fix_height(cube, plev_points_cube, plev_bounds_cube)
        lat_name = self.extra_facets.get('latitude', 'latitude')
        lon_name = self.extra_facets.get('longitude', 'longitude')
        if cube.coords(lat_name) and cube.coords(lon_name):
            self._fix_lat_lon(cube, lat_name, lon_name)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return iris.cube.CubeList([cube])

    def _fix_scalar_coords(self, cube):
        """Fix scalar coordinates."""
        if "height2m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if "height10m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if "typesi" in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

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

    @staticmethod
    def _fix_height(cube, plev_points_cube, plev_bounds_cube):
        """Fix height coordinate of cube."""
        air_pressure_points = plev_points_cube.core_data()

        # Get bounds from half levels and reshape array
        air_pressure_bounds = plev_bounds_cube.core_data()
        air_pressure_bounds = da.stack(
            (air_pressure_bounds[:, :-1], air_pressure_bounds[:, 1:]), axis=-1)

        # Setup air pressure coordinate with correct metadata and add to cube
        air_pressure_coord = iris.coords.AuxCoord(
            air_pressure_points,
            bounds=air_pressure_bounds,
            var_name='plev',
            standard_name='air_pressure',
            long_name='pressure',
            units=plev_points_cube.units,
            attributes={'positive': 'down'},
        )
        cube.add_aux_coord(air_pressure_coord, np.arange(cube.ndim))

        # Reverse entire cube along height axis so that index 0 is surface
        # level
        cube = iris.util.reverse(cube, 'height')

        # Fix metadata of generalized height coordinate
        z_coord = cube.coord('height')
        z_coord.var_name = 'model_level'
        z_coord.standard_name = None
        z_coord.long_name = 'model level number'
        z_coord.units = 'no unit'
        z_coord.attributes['positive'] = 'up'
        z_coord.points = np.arange(len(z_coord.points))
        z_coord.bounds = None

        return cube

    @staticmethod
    def _fix_lat_lon(cube, lat_name, lon_name):
        """Fix latitude and longitude coordinates of cube."""
        lat = cube.coord(lat_name)
        lon = cube.coord(lon_name)

        # Fix metadata
        lat.var_name = "lat"
        lon.var_name = "lon"
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat.long_name = "latitude"
        lon.long_name = "longitude"
        lat.convert_units('degrees_north')
        lon.convert_units('degrees_east')

        # If grid is not unstructured, no further changes are necessary
        if cube.coord_dims(lat) != cube.coord_dims(lon):
            return
        horizontal_coord_dims = cube.coord_dims(lat)
        if len(horizontal_coord_dims) != 1:
            return

        # Add dimension name for cell index used to store the unstructured grid
        index_coord = iris.coords.DimCoord(
            np.arange(cube.shape[horizontal_coord_dims[0]]),
            var_name="i",
            long_name=("first spatial index for variables stored on an "
                       "unstructured grid"),
            units="1",
        )
        cube.add_dim_coord(index_coord, horizontal_coord_dims)

    @staticmethod
    def _fix_time(cube):
        """Fix time coordinate of cube."""
        t_coord = cube.coord("time")
        t_coord.var_name = 'time'
        t_coord.standard_name = 'time'
        t_coord.long_name = 'time'

        # Convert invalid time units of the form "day as %Y%m%d.%f" to CF
        # format (e.g., "days since 1850-01-01")
        # Notes:
        # - It might be necessary to expand this to other time formats in the
        #   raw file.
        # - This has not been tested with sub-daily data
        time_format = 'day as %Y%m%d.%f'
        t_unit = t_coord.attributes.pop("invalid_units")
        if t_unit != time_format:
            raise ValueError(
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'")
        new_t_unit = cf_units.Unit('days since 1850-01-01',
                                   calendar="proleptic_gregorian")

        new_datetimes = [datetime.strptime(str(dt), '%Y%m%d.%f') for dt in
                         t_coord.points]
        new_dt_points = date2num(np.array(new_datetimes), new_t_unit)

        t_coord.points = new_dt_points
        t_coord.units = new_t_unit


class Siconca(Fix):
    """Fixes for ``siconca``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Note: This fix is called before the AllVars() fix. The wrong var_name
        # and units (which need to be %) are fixed in a later step in
        # AllVars(). This fix here is necessary to fix the "unknown" units that
        # cannot be converted to % in AllVars().
        cube = cubes.extract_cube(var_name_constraint('sic'))
        cube.units = '1'
        return cubes
