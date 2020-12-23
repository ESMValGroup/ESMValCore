"""Fixes for FIO-ESM-2-0 model."""
import iris

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix non-monotonic latitude coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords('latitude', dim_coords=False):
                lat_coord = cube.coord('latitude')
                lat_coord.bounds = None
                lat_coord.guess_bounds()
                iris.util.promote_aux_coord_to_dim_coord(cube, 'latitude')
        return cubes
