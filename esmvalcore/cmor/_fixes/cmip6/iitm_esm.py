"""Fixes for IITM-ESM model."""
from ..common import OceanFixGrid
from ..fix import Fix
import numpy as np
import cftime
from esmvalcore.iris_helpers import date2num

Tos = OceanFixGrid

class AllVars(Fix):
    """Fixes for all vars."""
    def fix_metadata(self, cubes):
        """Fix parent time units.

        IITM-ESM monthly data may have a bad time bounds spanning 20 days.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            if "mon" in cube.attributes['table_id']:
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