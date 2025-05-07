"""Fixes for obs4MIPs dataset C3S-GTO-ECV-9-0."""

import dask.array as da

from ..fix import Fix


class Toz(Fix):
    """Fixes for toz."""

    def fix_data(self, cube):
        """Fix data.

        Convert nan's to fill missing values.

        Parameters
        ----------
        cube: iris.cube
            Input cube.

        Returns
        -------
        iris.cube
            Fixed cube.

        """
        cube.data = da.ma.fix_invalid(cube.core_data())
        return cube
