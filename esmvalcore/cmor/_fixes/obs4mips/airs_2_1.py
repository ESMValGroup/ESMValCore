"""Fixes for obs4mips dataset AIRS-2-1."""
import iris
from iris.cube import CubeList
from cf_units import Unit

from ..fix import Fix


class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Change unit of coordinate plev from hPa to Pa

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        for cube in cubes:
            try:
                plev = cube.coord('air_pressure')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            else:
                plev.units = Unit('Pa')
        return cubes
