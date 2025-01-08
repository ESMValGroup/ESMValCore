"""Fixes for CanESM5 model."""
import dask.array as da
from iris.cube import CubeList

from ..fix import Fix

class Vt100(Fix):
    """Fixes for vt100."""

    def fix_metadata(self, cubes):
        """Convert units from W/m2 to K m/s.

        Parameters
        ----------
        cube : iris.cube.CubeListCubeList

        Returns
        -------
        iris.cube.CubeList

        """
        if not isinstance(cubes,CubeList):
            cubes = [cubes]
        for cube in cubes:
            if cube.units == "W m-2":
                cube.units = "K m s-1"
        return cubes

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
        cube *= 1.e-6
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
