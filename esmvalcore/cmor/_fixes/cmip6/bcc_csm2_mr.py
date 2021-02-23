"""Fixes for BCC-CSM2-MR model."""
from ..fix import Fix
from ..cmip5.bcc_csm1_1 import Tos as BaseTos
from ..common import ClFixHybridPressureCoord

import iris
import numpy as np

class allvars(Fix):
    """Common fixes to all vars"""

    def fix_metadata(self, cubes):
        """
        Fix metadata.
        Fixes error in time coordinate, sometimes contains trailing zeros
        Parameters
        ----------
        cube: iris.cube.CubeList
        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            try:
                old_time = cube.coord('time')
                if old_time.is_monotonic():
                    pass

                time_units = old_time.units
                time_data = old_time.points

                idx_zeros = np.where(time_data == 0.0)[0]
                time_diff = time_units.num2date(time_data[1]) \
                            - time_units.num2date(time_data[0])
                days = time_diff.days

                for idx in idx_zeros:
                    if idx == 0:
                        continue
                    correct_time = time_units.num2date(time_data[idx - 1])
                    if days <= 31 and days >=28:  # assume monthly time steps
                        if correct_time.month < 12:
                            new_time = \
                                correct_time.replace(month=correct_time.month + 1)
                        else:
                            new_time = \
                                correct_time.replace(year=correct_time.year + 1, month=1)
                    else:  # use "time[1] - time[0]" as step
                        new_time = correct_time + time_diff
                    old_time.points[idx] = time_units.date2num(new_time)

                # create new time bounds
                old_time.bounds = None
                old_time.guess_bounds()

                # replace time coordinate with "repaired" values
                new_time = iris.coords.DimCoord.from_coord(old_time)
                time_idx = cube.coord_dims(old_time)
                cube.remove_coord('time')
                cube.add_dim_coord(new_time, time_idx)

            except iris.exceptions.CoordinateNotFoundError:
                pass

        return cubes


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


class Tos(BaseTos):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of 1D-``latitude`` and 1D-``longitude``.
        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        lat_coord = cube.coord('latitude', dimensions=(1, ))
        lon_coord = cube.coord('longitude', dimensions=(2, ))
        lat_coord.standard_name = None
        lat_coord.long_name = 'grid_latitude'
        lat_coord.var_name = 'i'
        lat_coord.units = '1'
        lon_coord.standard_name = None
        lon_coord.long_name = 'grid_longitude'
        lon_coord.var_name = 'j'
        lon_coord.units = '1'
        lon_coord.circular = False
        return cubes


class Siconc(BaseTos):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of 1D-``latitude`` and 1D-``longitude``.
        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        lat_coord = cube.coord('latitude', dimensions=(1, ))
        lon_coord = cube.coord('longitude', dimensions=(2, ))
        lat_coord.standard_name = None
        lat_coord.long_name = 'grid_latitude'
        lat_coord.var_name = 'i'
        lat_coord.units = '1'
        lon_coord.standard_name = None
        lon_coord.long_name = 'grid_longitude'
        lon_coord.var_name = 'j'
        lon_coord.units = '1'
        lon_coord.circular = False
        return cubes


class Sos(Tos):
    """Fixes for sos."""
