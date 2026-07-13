"""Fixes for CanESM5 model."""

import dask.array as da

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.common import RenamePsStandardName

class AllVars(RenamePsStandardName):
    """Fixes for all variables, relevant only for variables with hybrid-sigma pressure levels."""

    
class Co2(Fix):
    """Fixes for co2."""

    def fix_data(self, cube):
        """Convert units from ppmv to 1.

        Parameters
        ----------
        cube : iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1.0e-6
        cube.metadata = metadata
        return cube


class Gpp(Fix):
    """Fixes for gpp, ocean values set to 0 instead of masked."""

    def fix_data(self, cube):
        """Fix masked values.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.0)
        return cube
