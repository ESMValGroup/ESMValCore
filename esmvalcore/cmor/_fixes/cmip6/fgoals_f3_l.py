"""Fixes for FGOALS-F3-L model."""
import numpy as np
from ..fix import Fix

class AllVars(Fix):
    """Fixes for all variables."""

    def fix_data(self, cube):
        """Fix data.

        Calculate missing latitude/longitude boundaries
        using contiguous_bounds method

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """

        xbounds = cube.coord('longitude').contiguous_bounds()
        ybounds = cube.coord('latitude').contiguous_bounds()
        xbnd = np.zeros((xbounds.size-1, 2))
        ybnd = np.zeros((ybounds.size-1, 2))
        xbnd[:, 0] = xbounds[0:-1]
        xbnd[:, 1] = xbounds[1:  ]
        ybnd[:, 0] = ybounds[0:-1]
        ybnd[:, 1] = ybounds[1:  ]
        cube.coord('longitude').bounds = xbnd
        cube.coord('latitude').bounds = ybnd

        return cube
