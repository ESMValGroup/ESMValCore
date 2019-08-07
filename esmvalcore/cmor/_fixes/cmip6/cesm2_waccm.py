"""Fixes for CESM2-WACCM model."""
import iris
import numpy as np

from .cesm2 import Tas as BaseTas
from ..fix import Fix


class Tas(BaseTas):
    """Fixes for tas."""


class Ua(Fix):
    """Fixes for ua."""

    def fix_metadata(self, cubes):
        """Fix non-monotonic time axis.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        dim_coords = [coord.name() for coord in cube.coords(dim_coords=True)]
        if 'time' in dim_coords:
            return cubes
        coord = cube.coord('time')
        idx_sorted = np.argsort(coord.points)
        coord.points = coord.points[idx_sorted]
        coord.bounds = None
        # coord.guess_bounds('time')  # fails for coordinate 'time'
        time_idx = cube.coord_dims('time')[0]
        slice_idx = [slice(None)] * cube.ndim
        slice_idx[time_idx] = idx_sorted
        cube.data = cube.data[slice_idx]
        iris.util.promote_aux_coord_to_dim_coord(cube, 'time')
        return cubes
