"""Fixes for rcm CNRM-ALADIN63 driven by CNRM-CERFACS-CNRM-CM5."""
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord

import numpy as np


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate and correct long_name for time.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube)
            if cube.coord('height').points != 2.:
                cube.coord('height').points = np.ma.array([2.0])
            cube.coord('time').long_name = 'time'

        return cubes


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """
        Correct long_name for time.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('time').long_name = 'time'

        return cubes
