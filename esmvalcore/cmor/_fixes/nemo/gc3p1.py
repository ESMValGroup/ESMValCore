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

#    def remove_extra_time_axis(cube):
#        count_time_axes = 0
#        time_axes_names = []
#        for coord in cube.coords():
#            if iris.util.guess_coord_axis(coord) == 'T':
#                count_time_axes += 1
#                time_axes_names.append(coord.standard_name)
#
#        if count_time_axes >= 2 and len(set(time_axes_names)) == 1:
#            for aux_coord in cube.coords(dim_coords=False):
#                if iris.util.guess_coord_axis(aux_coord) == 'T':
#                    cube.remove_coord(aux_coord)
#        else:
#            for coord in cube.coords():
#                if coord.var_name == 'time_counter':
#                    cube.remove_coord(coord)
