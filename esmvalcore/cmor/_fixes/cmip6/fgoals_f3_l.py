"""Fixes for CMIP6 FGOALS-f3-L model."""
import cftime
import dask.array as da
import numpy as np

from esmvalcore.iris_helpers import date2num

from ..common import OceanFixGrid
from ..fix import Fix

Tos = OceanFixGrid


Omon = OceanFixGrid


class AllVars(Fix):
    """Fixes for all vars."""
    def fix_metadata(self, cubes):
        """Fix parent time units.

        FGOALS-f3-L Amon data may have a bad time bounds spanning 20 days.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            if cube.attributes['table_id'] == 'Amon':
                for coord in ['latitude', 'longitude']:
                    cube_coord = cube.coord(coord)
                    bounds = cube_coord.bounds
                    if np.any(bounds[:-1, 1] != bounds[1:, 0]):
                        cube_coord.bounds = None
                        cube_coord.guess_bounds()
                time = cube.coord('time')
                if np.any(time.bounds[:-1, 1] != time.bounds[1:, 0]):
                    times = time.units.num2date(time.points)
                    starts = [
                        cftime.DatetimeNoLeap(c.year, c.month, 1)
                        for c in times
                    ]
                    ends = [
                        cftime.DatetimeNoLeap(c.year, c.month +
                                              1, 1) if c.month < 12 else
                        cftime.DatetimeNoLeap(c.year + 1, 1, 1) for c in times
                    ]
                    time.bounds = date2num(np.stack([starts, ends], -1),
                                           time.units)
        return cubes


class Sftlf(Fix):
    """Fixes for sftlf."""
    def fix_data(self, cube):
        """Fix data.

        Unit is %, values are <= 1.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix

        Returns
        -------
        iris.cube.Cube
            Fixed cube. It can be a difference instance.
        """
        if cube.units == "%" and da.max(cube.core_data()).compute() <= 1.:
            cube.data = cube.core_data() * 100.
        return cube
