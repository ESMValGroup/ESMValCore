"""Fixes for CESM1-BGC model."""

from dask import array as da

from ..fix import Fix


class Cl(Fix):
    """Fixes for cl."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


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
