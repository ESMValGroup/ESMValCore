import cf_units
from datetime import datetime
import iris
import numpy as np

import logging
from ..fix import Fix
from ..shared import add_scalar_height_coord


logger = logging.getLogger(__name__)


class AllVars(Fix):

    # TODO Read this from file
    TRANSLATION_TABLE = {"cell_area": "areacella"}

    def fix_metadata(self, cubes):
        # logger.info(f"\nFix metadata\n==================\n{cubes}\n")
        for cube in cubes:
            # Fixing varnames
            if cube.var_name in self.TRANSLATION_TABLE:
                cube.var_name = self.TRANSLATION_TABLE[cube.var_name]
            if cube.var_name == self.vardef.short_name:
                logger.info(f"Fixing {cube.var_name}")
                if cube.coords("time"):
                    self._fix_time(cube)
                if cube.coords("latitude") and cube.coords("longitude"):
                    self._fix_coordinates(cube, "latitude", "longitude")
                elif cube.coords("grid_latitude") and cube.coords("grid_longitude"):
                    self._fix_coordinates(cube, "grid_latitude", "grid_longitude")
        return cubes

    def _fix_coordinates(self, cube, lat_name, lon_name):
        lat = cube.coord(lat_name)
        lon = cube.coord(lon_name)
        # logger.info(f"{cube.var_name} : lat={lat}, lon={lon}")
        lat.var_name = "lat"
        lon.var_name = "lon"

        # Only necessary for Areacella so far
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat.long_name = "latitude"
        lon.long_name = "longitude"

        if cube.coords("time"):
            spatial_index = 1
        else:
            spatial_index = 0

        index_coord = iris.coords.DimCoord(
            np.arange(cube.shape[spatial_index]),
            var_name="i",
            long_name="first spatial index for variables stored on an unstructured grid",
            units="1",
        )
        cube.add_dim_coord(index_coord, spatial_index)

        if "height2m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if "height10m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)

    @staticmethod
    def _fix_time(cube):
        t_coord = cube.coord("time")

        t_unit = t_coord.attributes["invalid_units"]
        timestep, _, t_fmt_str = t_unit.split(" ")
        new_t_unit_str = f"{timestep} since 1850-01-01"
        new_t_unit = cf_units.Unit(new_t_unit_str, calendar="standard")

        new_datetimes = [datetime.strptime(str(dt), t_fmt_str) for dt in t_coord.points]
        new_dt_points = [new_t_unit.date2num(new_dt) for new_dt in new_datetimes]

        t_coord.points = new_dt_points
        t_coord.units = new_t_unit
