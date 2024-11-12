"""Fixes for CESM2-WACCM model."""

import iris
import numpy as np
from netCDF4 import Dataset

from ..common import SiconcFixScalarCoord
from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Tas as BaseTas


class Cl(BaseCl):
    """Fixes for cl."""

    def fix_file(self, filepath, output_dir, add_unique_suffix=False):
        """Fix hybrid pressure coordinate.

        Adds missing ``formula_terms`` attribute to file.

        Note
        ----
        Fixing this with :mod:`iris` in ``fix_metadata`` or ``fix_data`` is
        **not** possible, since the bounds of the vertical coordinates ``a``
        and ``b`` are not present in the loaded :class:`iris.cube.CubeList`,
        even when :func:`iris.load_raw` is used.

        Parameters
        ----------
        filepath : str
            Path to the original file.
        output_dir: Path
            Output directory for fixed files.
        add_unique_suffix: bool, optional (default: False)
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        str
            Path to the fixed file.

        """
        new_path = self._fix_formula_terms(
            filepath, output_dir, add_unique_suffix=add_unique_suffix
        )
        dataset = Dataset(new_path, mode="a")
        dataset.variables["a_bnds"][:] = dataset.variables["a_bnds"][:, ::-1]
        dataset.variables["b_bnds"][:] = dataset.variables["b_bnds"][:, ::-1]
        dataset.close()
        return new_path


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Tas = BaseTas


class Tasmin(Tas):
    """Fixes for tasmin."""

    def fix_metadata(self, cubes):
        """Fix time coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
                Cubes to fix

        Returns
        -------
        iris.cube.CubeList

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
