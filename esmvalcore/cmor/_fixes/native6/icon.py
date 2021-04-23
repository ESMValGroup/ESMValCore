import cf_units
from datetime import datetime
import iris
import numpy as np

# from netCDF4 import Dataset

import logging
from ..fix import Fix
from ..shared import add_scalar_height_coord


logger = logging.getLogger(__name__)

# TODO Obtain this information from the dataset keys
GRID_FILE = "/mnt/lustre02/work/bd1179/experiments/icon-2.6.1_atm_amip_R2B5_r1v1i1p1l1f1"


class AllVars(Fix):

    def fix_metadata(self, cubes):
        logger.info(f"\nFix metadata\n==================\n{cubes}\n")
        logger.info(f"self.vardef = {self.vardef}")
        for cube in cubes:
            logger.info(f"Fixing {cube.var_name}")
            self._fix_time(cube)
            self._fix_coordinates(cube)
        return cubes

    
    def _fix_coordinates(self, cube):
        lon = cube.coord("longitude")
        lat = cube.coord("latitude")
#         logger.info(f"{cube.var_name} : lat={lat}, lon={lon}")
        # TODO
        lon.var_name = "lon"
        lat.var_name = "lat"
        logger.info(f"Before Manuel: {cube.var_name} : {cube}")
        
        index_coord = iris.coords.DimCoord(
            np.arange(cube.shape[1]),
            var_name="i",
            long_name="first spatial index for variables stored on an unstructured grid",
            units="1",
        )
        cube.add_dim_coord(index_coord, 1)
        logger.info(f"After Manuel: {cube.var_name} : {cube}")
        
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.)
            

    @staticmethod
    def _fix_time(cube):
        t_coord = cube.coord("time")
        
        t_unit = t_coord.attributes["invalid_units"]
        timestep, _, t_fmt_str = t_unit.split(" ")
        new_t_unit_str = f"{timestep} since 1850-01-01"
        new_t_unit = cf_units.Unit(new_t_unit_str, calendar="standard")
        
        new_datetimes = [datetime.strptime(str(dt), t_fmt_str)
                         for dt in t_coord.points]
        new_dt_points = [new_t_unit.date2num(new_dt) for new_dt in new_datetimes]
        
        t_coord.points = new_dt_points
        t_coord.units = new_t_unit
        
        
        
        