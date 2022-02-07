"""Fixes for obs4MIPs dataset AIRS-2-1."""
import iris
from cf_units import Unit
from iris.cube import CubeList

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
                if plev.points[0] > 10000.0:
                    plev.units = Unit('Pa')
        return cubes
