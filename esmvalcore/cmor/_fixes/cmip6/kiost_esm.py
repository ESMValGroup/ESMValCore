"""Fixes for KIOST-ESM model."""
from dask import array as da

from ..common import SiconcFixScalarCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cubes
            Fixed cubes.
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return cubes


class Hurs(Tas):
    """Fixes for hurs."""


class Huss(Tas):
    """Fixes for huss."""


class Tasmin(Tas):
    """Fixes for tasmin."""


class Tasmax(Tas):
    """Fixes for tasmax."""


class SfcWind(Fix):
    """Fixes for sfcWind."""

    def fix_metadata(self, cubes):
        """Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cubes
            Fixed cubes.
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


class Uas(SfcWind):
    """Fixes for uas."""


class Vas(SfcWind):
    """Fixes for vas."""


class Siconc(SiconcFixScalarCoord):
    """Fixes for siconc."""

    def fix_data(self, cube):
        """Fix data.

        Fix missing values.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube
            Fixed cube.
        """
        cube.data = da.ma.masked_invalid(cube.core_data())
        return cube
