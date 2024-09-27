"""Fixes for FGOALS-g3 model."""

import dask.array as da
import iris

from ..common import OceanFixGrid
from ..fix import Fix


def _check_bounds_monotonicity(coord):
    """Check monotonicity of a coords bounds array."""
    if coord.has_bounds():
        for i in range(coord.nbounds):
            if not iris.util.monotonic(coord.bounds[..., i], strict=True):
                return False

    return True


class Tos(OceanFixGrid):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        FGOALS-g3 data contain latitude and longitude data set to >1e30 in some
        places. Note that the corresponding data is all masked.

        Example files:
        v20191030/siconc_SImon_FGOALS-g3_piControl_r1i1p1f1_gn_070001-079912.nc
        v20191030/siconc_SImon_FGOALS-g3_piControl_r1i1p1f1_gn_080001-089912.nc
        v20200706/siconc_SImon_FGOALS-g3_ssp534-over_r1i1p1f1_gn_201501-210012
        .nc

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        for coord_name in ["latitude", "longitude"]:
            coord = cube.coord(coord_name)
            bad_indices = coord.core_points() > 1000.0
            coord.points = da.ma.where(bad_indices, 0.0, coord.core_points())

        return super().fix_metadata(cubes)


Siconc = Tos


class Mrsos(Fix):
    """Fixes for mrsos."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        FGOALS-g3 mrsos data contains error in coordinate bounds.

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
