"""Fixes for CESM2-WACCM model."""
from netCDF4 import Dataset

from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas

from ..fix import Fix

import numpy as np
import iris
import cf_units


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


class msftmz(Fix):
    """Fix msftmz."""

    def fix_metadata(self, cubes):
        """
        Problems:
         basin has incorrect long name, var.
         Dimensions are also wrong.
        Parameters
        ----------
        cube: iris.cube.CubeList
        Returns
        -------
        iris.cube.CubeList
        """
        new_cubes = []
        for cube in cubes:

            # Fix regions coordinate
            cube.remove_coord(cube.coord("region"))
            values = np.array(['atlantic_arctic_ocean', 'indian_pacific_ocean',
                               'global_ocean',], dtype='<U21')
            basin_coord = iris.coords.AuxCoord(
                values,
                standard_name=u'region',
                units=cf_units.Unit('no_unit'),
                long_name=u'ocean basin',
                var_name='basin')

            # Replace broken coord with correct one.
            cube.add_aux_coord(basin_coord, data_dims=1)
#            # Fix depth
            depth = cube.coord('lev')
            depth.var_name = 'depth'
            depth.standard_name = 'depth'
            depth.long_name = 'depth'
        return cubes

