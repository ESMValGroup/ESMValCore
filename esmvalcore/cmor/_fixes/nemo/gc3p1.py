"""Fixes for GC3.1 data"""

from esmvalcore.cmor.fix import Fix
from iris.util import promote_aux_coord_to_dim_coord


class AllVars(Fix):
    """"""

    def fix_metadata(self, cubes):
        """"""
        # Replace time_counter with time_centered/time_instant
        for cube in cubes:
            for coordinate in cube.coords(dim_coords=True):
                if coordinate.var_name == 'time_counter':
                    cube.remove_coord(coordinate)
                    promote_aux_coord_to_dim_coord(cube, 'time')
                    cube.coord('time').var_name = 'time'
                    break

        return cubes
