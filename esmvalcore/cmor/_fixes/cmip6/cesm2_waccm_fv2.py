"""Fixes for CESM2-WACCM-FV2 model."""

from ..common import SiconcFixScalarCoord
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Tas as BaseTas
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw
from ..fix import Fix
import numpy as np
import iris

Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Tas = BaseTas

class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """Fix time coordinates.

        Parameters
        ----------
        cubes : iris.cube.CubeList
                Cubes to fix

        Returns
        -------
        iris.cube.CubeList, iris.cube.CubeList
        """
        new_list = iris.cube.CubeList()
        for cube in cubes:
            try:
                old_time = cube.coord("time")
            except iris.exceptions.CoordinateNotFoundError:
                new_list.append(cube)
            else:
                if old_time.is_monotonic():
                    new_list.append(cube)
                else:
                    time_units = old_time.units
                    time_data = old_time.points

                    # erase erroneously copy-pasted points
                    time_diff = np.diff(time_data)
                    idx_neg = np.where(time_diff <= 0.0)[0]
                    while len(idx_neg) > 0:
                        time_data = np.delete(time_data, idx_neg[0] + 1)
                        time_diff = np.diff(time_data)
                        idx_neg = np.where(time_diff <= 0.0)[0]

                    # create the new time coord
                    new_time = iris.coords.DimCoord(
                        time_data,
                        standard_name="time",
                        var_name="time",
                        units=time_units,
                    )

                    # create a new cube with the right shape
                    dims = (
                        time_data.shape[0],
                        cube.coord("latitude").shape[0],
                        cube.coord("longitude").shape[0],
                    )
                    data = cube.data
                    new_data = np.ma.append(
                        data[: dims[0] - 1, :, :], data[-1, :, :]
                    )
                    new_data = new_data.reshape(dims)

                    tmp_cube = iris.cube.Cube(
                        new_data,
                        standard_name=cube.standard_name,
                        long_name=cube.long_name,
                        var_name=cube.var_name,
                        units=cube.units,
                        attributes=cube.attributes,
                        cell_methods=cube.cell_methods,
                        dim_coords_and_dims=[
                            (new_time, 0),
                            (cube.coord("latitude"), 1),
                            (cube.coord("longitude"), 2),
                        ],
                    )

                    new_list.append(tmp_cube)
        return new_list