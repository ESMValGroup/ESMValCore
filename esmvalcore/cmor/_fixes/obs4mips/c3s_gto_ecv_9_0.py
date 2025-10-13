"""Fixes for obs4MIPs dataset C3S-GTO-ECV-9-0."""

import dask.array as da
from iris.cube import Cube

from esmvalcore.cmor._fixes.fix import Fix


class Toz(Fix):
    """Fixes for toz."""

    def fix_data(self, cube: Cube) -> Cube:
        """Fix data.

        Mask nan's.

        Parameters
        ----------
        cube:
            Input cube.

        Returns
        -------
        iris.cube.Cube
            Fixed cube.

        """
        cube.data = da.ma.fix_invalid(cube.core_data())
        return cube
