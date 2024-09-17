"""Fixes for FGOALS-s2 model."""
import iris

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fix wrong bounds of latitude coordinate at first and last index.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            try:
                lat_coord = cube.coord('latitude')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            if lat_coord.ndim != 1:
                continue
            if lat_coord.shape[0] < 3:
                continue
            lat_bounds = lat_coord.core_bounds().copy()
            lat_diff = lat_bounds[1][1] - lat_bounds[1][0]
            lat_bounds[0][0] = lat_bounds[0][1] - lat_diff
            lat_bounds[-1][1] = lat_bounds[-1][0] + lat_diff
            lat_coord.bounds = lat_bounds
        return cubes
