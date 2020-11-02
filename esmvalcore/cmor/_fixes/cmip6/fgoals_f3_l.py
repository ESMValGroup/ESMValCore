"""Fixes for CMIP6 FGOALS-f3-L."""
import cftime
import numpy as np

from ..fix import Fix


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
                    time.bounds = time.units.date2num(
                        np.stack([starts, ends], -1))
        return cubes
