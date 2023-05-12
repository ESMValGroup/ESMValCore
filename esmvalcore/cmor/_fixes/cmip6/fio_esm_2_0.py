"""Fixes for FIO-ESM-2-0 model."""
import iris

from ..common import OceanFixGrid
from ..fix import Fix
from ..shared import round_coordinates

Tos = OceanFixGrid

def _check_bounds_monotonicity(coord):
    """Check monotonicity of a coords bounds array."""
    if coord.has_bounds():
        for i in range(coord.nbounds):
            if not iris.util.monotonic(coord.bounds[..., i], strict=True):
                return False

    return True

class Omon(Fix):
    """Fixes for Omon vars."""

    def fix_metadata(self, cubes):
        """
        Fix latitude and longitude with round to 6 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        round_coordinates(cubes, decimals=6, coord_names=["longitude", "latitude"])
        return cubes


class Amon(Fix):
    """Fixes for Amon vars."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        FIO-ESM-2-0 ta data contains error in co-ordinate bounds.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)

        # Check both lat and lon coords and replace bounds if necessary
        if not _check_bounds_monotonicity(cube.coord("latitude")):
            cube.coord("latitude").bounds = None
            cube.coord("latitude").guess_bounds()
            iris.util.promote_aux_coord_to_dim_coord(cube, "latitude")

        if not _check_bounds_monotonicity(cube.coord("longitude")):
            cube.coord("longitude").bounds = None
            cube.coord("longitude").guess_bounds()
            iris.util.promote_aux_coord_to_dim_coord(cube, "longitude")

        return super().fix_metadata(cubes)
