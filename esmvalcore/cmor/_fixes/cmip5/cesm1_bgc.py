"""Fixes for CESM1-BGC model."""

from dask import array as da

from ..fix import Fix
from .cesm1_cam5 import Cl as BaseCl


Cl = BaseCl


class Gpp(Fix):
    """Fixes for gpp variable."""

    def fix_data(self, cube):
        """Fix data.

        Fix missing values.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        data = da.ma.masked_equal(cube.core_data(), 1.0e33)
        return cube.copy(data)


class Nbp(Gpp):
    """Fixes for nbp variable."""
