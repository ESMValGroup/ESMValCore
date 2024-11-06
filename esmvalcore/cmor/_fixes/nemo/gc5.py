"""Fixes for GC5 data"""

from esmvalcore.cmor.fix import Fix
from iris.util import promote_aux_coord_to_dim_coord


class AllVars(Fix):
    """"""

    def fix_metadata(self, cubes):
        """"""
        # Correct variable name
        for cube in cubes:
            if cube.var_name.endswith('_con'):
                cube.var_name = cube.var_name[:-4]

            for coordinate in cube.coords(dim_coords=True):
                if coordinate.var_name == 'time_counter':
                    cube.remove_coord(coordinate)
                    promote_aux_coord_to_dim_coord(cube, 'time')
                    cube.coord('time').var_name = 'time'
                    break

        return cubes

#check with emma about adding in the fix
