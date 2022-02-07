"""Fixes for obs4MIPs dataset AIRS-2-1."""
from iris.exceptions import CoordinateNotFoundError

from ..fix import Fix


class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Change unit of coordinate plev from hPa to Pa.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes.

        """
        for cube in cubes:
            try:
                plev = cube.coord('air_pressure')
            except CoordinateNotFoundError:
                continue
            else:
                if plev.points[0] > 10000.0:
                    plev.units = 'Pa'
        return cubes
