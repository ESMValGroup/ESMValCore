"""Fixes for CESM2-WACCM model."""
from netCDF4 import Dataset

from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas

from ..fix import Fix
import numpy as np
import iris

class Cl(BaseCl):
    """Fixes for cl."""

    def fix_file(self, filepath, output_dir):
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
        output_dir : str
            Path of the directory where the fixed file is saved to.

        Returns
        -------
        str
            Path to the fixed file.

        """
        new_path = self._fix_formula_terms(filepath, output_dir)
        dataset = Dataset(new_path, mode='a')
        dataset.variables['a_bnds'][:] = dataset.variables['a_bnds'][:, ::-1]
        dataset.variables['b_bnds'][:] = dataset.variables['b_bnds'][:, ::-1]
        dataset.close()
        return new_path


Cli = Cl


Clw = Cl


Tas = BaseTas


class AllVars(Fix):
    """Fixes for all vars."""
    def fix_metadata(self, cubes):
        """Fix daily timecoord and bounds.

        Issue
        -----
        Bounds might be wrong by one day
        DimCoord([2010-01-01 00:00:00, 2010-01-02 00:00:00, ...],
            bounds=[[2009-12-31 00:00:00, 2010-01-01 00:00:00],
                    [2010-01-01 00:00:00, 2010-01-02 00:00:00], ...], ...)
        For SSPs bounds of 2015-01-01 violate strictly monotonic rule:
            bounds=[[2015-01-01 00:00:00, 2015-01-01 00:00:00],..]
            leading to time coordinate treated as aux coord

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.attributes['mipTable'] == 'day':
            # coorect time coord points and bounds
            time = cube.coord('time')
            times = time.units.num2date(time.points)
            if np.all(np.array([c.hour for c in times]) == 0):
                time.points = time.points + 0.5
                time.bounds = None
                time.guess_bounds()
            # set time to dim_coord
            if time not in cube.coords(dim_coords=True):
                iris.util.promote_aux_coord_to_dim_coord(cube, 'time')
        return cubes
