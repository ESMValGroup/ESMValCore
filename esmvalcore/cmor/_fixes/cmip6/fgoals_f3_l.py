"""Fixes for CMIP6 FGOALS-f3-L model."""
import cftime
import dask.array as da
import numpy as np

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all vars."""

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
        cube.coord('latitude').bounds = None
        cube.coord('longitude').bounds = None
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
