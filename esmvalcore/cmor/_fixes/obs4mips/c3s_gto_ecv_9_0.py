"""Fixes for obs4MIPs dataset C3S-GTO-ECV-9-0."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da

from esmvalcore.cmor._fixes.fix import Fix

if TYPE_CHECKING:
    from iris.cube import Cube


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
