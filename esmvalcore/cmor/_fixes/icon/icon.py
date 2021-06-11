"""On-the-fly CMORizer for ICON."""

import logging
from datetime import datetime

import cf_units
import dask.array as da
import iris
import numpy as np

from esmvalcore.iris_helpers import var_name_constraint
from esmvalcore.preprocessor import regrid
from ..fix import Fix
from ..shared import add_scalar_height_coord

logger = logging.getLogger(__name__)


class AllVars(Fix):
    """Fixes for all variables."""

    # TODO Read this from file
    # If variable is not in the table, raw_name == var_name
    TRANSLATION_TABLE = {
        'areacella': 'cell_area',
        'clwvi': 'cllvi',
        'siconca': 'sic',
    }

    def fix_metadata(self, cubes):
        """Fix metadata."""
        raw_name = self.TRANSLATION_TABLE.get(self.vardef.short_name,
                                              self.vardef.short_name)
        cube = cubes.extract_cube(var_name_constraint(raw_name))

        # Horizontal regridding (this is necessary here since the
        # preprocessor function 'regrid' does not work for ICON's 3D
        # variables: the necessary scheme 'unstructured' does not support
        # the hybrid pressure levels)
        # TODO: make this configurable in the recipe, e.g. by supporting
        # a variable key 'target_grid' (that could also be 'native') and
        # 'scheme'.
        target_grid = 'native'
        scheme = 'unstructured_nearest'
        cube = self._regrid(cube, target_grid, scheme)

        # Fix dimensional coordinates
        if cube.coords("time"):
            self._fix_time(cube)
        if cube.coords("height"):
            plev_points_cube = cubes.extract_cube(var_name_constraint(
                'pfull'))
            plev_bounds_cube = cubes.extract_cube(var_name_constraint(
                'phalf'))
            plev_points_cube = self._regrid(plev_points_cube, target_grid,
                                            scheme)
            plev_bounds_cube = self._regrid(plev_bounds_cube, target_grid,
                                            scheme)
            cube = self._fix_height(cube, plev_points_cube,
                                    plev_bounds_cube)
        if cube.coords("latitude") and cube.coords("longitude"):
            self._fix_lat_lon(cube, "latitude", "longitude")
        elif (cube.coords("grid_latitude") and
              cube.coords("grid_longitude")):
            self._fix_lat_lon(cube, "grid_latitude", "grid_longitude")

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
            # TODO: use general function for this (see #1105)
            typesi_coord = iris.coords.AuxCoord(
                'sea_ice',
                var_name='type',
                standard_name='area_type',
                long_name='Sea Ice area type',
                units='no unit',
            )
            if not cube.coords('area_type'):
                cube.add_aux_coord(typesi_coord, ())

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

        # Add 'positive' attribute to height so iris can determine Z axis
        # correctly
        cube.coord('height').attributes['positive'] = 'down'

        # Reverse height axis so that index 0 is surface level
        cube = iris.util.reverse(cube, 'height')
        return cube

    @staticmethod
    def _fix_lat_lon(cube, lat_name, lon_name):
        """Fix latitude and longitude coordinates of cube."""
        lat = cube.coord(lat_name)
        lon = cube.coord(lon_name)
        lat.var_name = "lat"
        lon.var_name = "lon"

        # Only necessary for areacella so far
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat.long_name = "latitude"
        lon.long_name = "longitude"

    @staticmethod
    def _fix_time(cube):
        """Fix time coordinate of cube."""
        t_coord = cube.coord("time")

        # Convert units to CF format
        t_unit = t_coord.attributes["invalid_units"]
        timestep, _, t_fmt_str = t_unit.split(" ")
        new_t_unit_str = f"{timestep} since 1850-01-01"
        new_t_unit = cf_units.Unit(new_t_unit_str, calendar="standard")

        new_datetimes = [datetime.strptime(str(dt), t_fmt_str) for dt in
                         t_coord.points]
        new_dt_points = [new_t_unit.date2num(new_dt) for new_dt in
                         new_datetimes]

        t_coord.points = new_dt_points
        t_coord.units = new_t_unit

    @staticmethod
    def _regrid(cube, target_grid, scheme):
        """Regrid cube."""
        # Check if file is already regridded (i.e., 1D lat and lon dimensions
        # are present)
        if all([cube.coords('latitude', dim_coords=True),
                cube.coords('longitude', dim_coords=True)]):
            for coord_name in ('latitude', 'longitude'):
                if cube.coord(coord_name).bounds is None:
                    cube.coord(coord_name).guess_bounds()
            return cube

        # If native grid is requested add coordinate that specifies cell index
        if target_grid == 'native':
            spatial_index = 0
            if cube.coords("time"):
                spatial_index += 1
            if cube.coords("height"):
                spatial_index += 1
            index_coord = iris.coords.DimCoord(
                np.arange(cube.shape[spatial_index]),
                var_name="i",
                long_name=("first spatial index for variables stored on an "
                           "unstructured grid"),
                units="1",
            )
            cube.add_dim_coord(index_coord, spatial_index)
            return cube

        # Regrid to desired grid using desired scheme
        return regrid(cube, target_grid, scheme, lon_offset=False)


class Siconca(Fix):
    """Fixes for ``siconca``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Note: wrong var_name gets fixed in AllVars
        cube = cubes.extract_cube(var_name_constraint('sic'))
        cube.units = '1'
        return cubes
