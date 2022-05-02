"""Fixes for GFDL-CM4 model."""
import iris

from ..common import ClFixHybridPressureCoord, SiconcFixScalarCoord
from ..fix import Fix
from ..shared import add_aux_coords_from_cubes, add_scalar_height_coord
from .gfdl_esm4 import Omon as BaseOmon
from .gfdl_esm4 import Fgco2 as BaseFgco2


class Cl(ClFixHybridPressureCoord):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix hybrid sigma pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        coords_to_add = {
            'ap': 1,
            'b': 1,
            'ps': (0, 2, 3),
        }
        add_aux_coords_from_cubes(cube, cubes, coords_to_add)
        return super().fix_metadata(cubes)


Cli = Cl


Clw = Cl


Siconc = SiconcFixScalarCoord


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        try:
            cube.coord('height')
        except iris.exceptions.CoordinateNotFoundError:
            add_scalar_height_coord(cube, 2.0)
        return cubes


class Uas(Fix):
    """Fixes for uas."""

    def fix_metadata(self, cubes):
        """
        Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


class Vas(Fix):
    """Fixes for vas."""

    def fix_metadata(self, cubes):
        """
        Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


Omon = BaseOmon


Oyr = Omon


Fgco2 = BaseFgco2
