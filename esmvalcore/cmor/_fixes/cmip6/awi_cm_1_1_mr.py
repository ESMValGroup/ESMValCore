"""Fixes for AWI-CM-1-1-MR model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord

def set_range_in_0_360(array):
    while array.min() < 0:
        array[array < 0] += 360
    while array.max() > 360:
        array[array > 360] -= 360
    return array

class Thetao(Fix):

    def fix_metadata(self, cubes):
        cube = self.get_cube_from_list(cubes)
        coord = cube.coord('longitude')
        new_lons = coord.points.copy()
        new_lons = set_range_in_0_360(new_lons)

        new_bounds = coord.bounds.copy()
        new_bounds = set_range_in_0_360(new_bounds)

        new_coord = coord.copy(new_lons, new_bounds)

        dims = cube.coord_dims(coord)
        cube.remove_coord(coord)
        cube.add_aux_coord(new_coord, dims)

        return cubes

So = Thetao

class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Rename ``long_name`` of latitude to latitude (may be Latitude).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube
        """
        coords_longnames_to_change = {
            'latitude': 'latitude',
        }

        for cube in cubes:
            for (std_name, long_name) in coords_longnames_to_change.items():
                coord = cube.coord(std_name)
                if coord.long_name != long_name:
                    coord.long_name = long_name

        return cubes
