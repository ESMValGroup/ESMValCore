# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for BCC-CSM2-MR."""
import numpy as np
import iris
from iris.coords import DimCoord

from ..fix import Fix


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
            except iris.exceptions.CoordinateNotFoundError:
                continue

            # if time variable is monotonic, there is nothing to do

            if old_time.is_monotonic():
                continue

            time_units = old_time.units
            time_data = old_time.points.copy()

            idx_zeros = np.where(time_data == 0.0)[0]
            time_diff = (time_units.num2date(time_data[1]) -
                         time_units.num2date(time_data[0]))
            days = time_diff.days

            for idx in idx_zeros:
                if idx == 0:
                    continue
                correct_time = time_units.num2date(time_data[idx - 1])
                if days <= 31 and days >= 28:  # assume monthly time steps
                    new_month = correct_time.month + 1
                    new_year = correct_time.year
                    if new_month > 12:
                        new_month = 1
                        new_year = new_year + 1
                    new_time = correct_time.replace(month=new_month,
                                                    year=new_year)
                else:  # use "time[1] - time[0]" as step
                    new_time = correct_time + time_diff
                time_data[idx] = time_units.date2num(new_time)

            # create new time variable with fixed data points
            new_time = DimCoord(time_data,
                                standard_name=old_time.standard_name,
                                long_name=old_time.long_name,
                                var_name=old_time.var_name,
                                units=old_time.units,
                                bounds=None,
                                attributes=old_time.attributes,
                                coord_system=old_time.coord_system)

            # create new time bounds
            new_time.guess_bounds()

            # replace time coordinate with repaired values
            time_idx = cube.coord_dims(old_time)
            cube.remove_coord('time')
            cube.add_dim_coord(new_time, time_idx)

        return cubes
