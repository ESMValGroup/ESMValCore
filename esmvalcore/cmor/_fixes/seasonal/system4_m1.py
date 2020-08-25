"""Fixes for System4-m1."""
import iris.exceptions
from iris.util import promote_aux_coord_to_dim_coord

from ..fix import Fix
from ..shared import add_aux_coords_from_cubes


class AllVars(Fix):
    """Fixes for clt."""

    def fix_metadata(self, cubes):
        if cubes.extract('realization'):
            add_aux_coords_from_cubes(
                cubes.extract('tos')[0], cubes, {'realization': 1}
            )

        for cube in cubes:
            try:
                lat = cube.coord('latitude')
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                lat.var_name = 'lat'
            try:
                lon = cube.coord('longitude')
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                lon.var_name = 'lon'
            try:
                ensemble = cube.coord('realization')
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                ensemble.var_name = 'ensemble'
                ensemble.long_name = 'ensemble'
                promote_aux_coord_to_dim_coord(cube, 'ensemble')

        return cubes
