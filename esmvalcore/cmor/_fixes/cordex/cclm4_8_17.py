"""Fixes for CCLM4-8-17."""
import numpy as np

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""
    def fix_metadata(self, cubes):
        """Fix datatype

        The data is stored as '>f4' which causes issues

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.data = cube.data.astype(np.float32, copy=False)

             # also modify dtype of time coords (stored as '>f8' for some...)
            cube.coord('time').points = cube.coord('time').points.astype(np.float64, copy=False)
            cube.coord('time').bounds = cube.coord('time').bounds.astype(np.float64, copy=False)

        return cubes
