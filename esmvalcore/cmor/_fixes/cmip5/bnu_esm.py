"""Fixes for BNU-ESM model."""

from cf_units import Unit
from dask import array as da

from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix


class Cl(ClFixHybridPressureCoord):
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


class FgCo2(Fix):
    """Fixes for fgco2."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes cube units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes to fix.

        Returns
        -------
        iris.cube.CubeList

        """
        self.get_cube_from_list(cubes).units = Unit("kg m-2 s-1")
        return cubes

    def fix_data(self, cube):
        """
        Fix data.

        Fixes cube units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 12.0 / 44.0
        cube.metadata = metadata
        return cube


class Ch4(Fix):
    """Fixes for ch4."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes cube units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes to fix.

        Returns
        -------
        iris.cube.CubeList

        """
        self.get_cube_from_list(cubes).units = Unit("1e-9")
        return cubes

    def fix_data(self, cube):
        """
        Fix metadata.

        Fixes cube units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.


        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 29.0 / 16.0 * 1.0e9
        cube.metadata = metadata
        return cube


class Co2(Fix):
    """Fixes for co2."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes cube units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes to fix.

        Returns
        -------
        iris.cube.CubeList

        """
        self.get_cube_from_list(cubes).units = Unit("1e-6")
        return cubes

    def fix_data(self, cube):
        """
        Fix data.

        Fixes cube units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 29.0 / 44.0 * 1.0e6
        cube.metadata = metadata
        return cube


class SpCo2(Fix):
    """Fixes for spco2."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes cube units.

        Parameters
        ----------
        cube : iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1.0e6
        cube.metadata = metadata
        return cube


class Od550Aer(Fix):
    """Fixes for od550aer."""

    def fix_data(self, cube):
        """
        Fix data.

        Masks invalid values.

        Parameters
        ----------
        cube : iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        data = da.ma.masked_equal(cube.core_data(), 1.0e36)
        return cube.copy(data)
