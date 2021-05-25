"""Fixes for CESM2-WACCM model."""
from netCDF4 import Dataset

from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas
from ..fix import Fix
from ..shared import fix_ocean_depth_coord


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


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords('latitude'):
                cube.coord('latitude').var_name = 'lat'
            if cube.coords('longitude'):
                cube.coord('longitude').var_name = 'lon'

            if cube.coords(axis='Z'):
                if str(z.coords.units) == 'cm' and np.max(z.points)>10000.:
                    z_coord.units = cf_units.Unit('m')
                fix_ocean_depth_coord(cube)
        return cubes


