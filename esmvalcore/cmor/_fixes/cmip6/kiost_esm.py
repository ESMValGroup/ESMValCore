"""Fixes for KIOST-ESM model."""
import iris

from ..fix import Fix
from ..shared import add_scalar_height_coord

class Hurs(Fix):
    """Fixes for hurs."""
    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return cubes



class Huss(Fix):
    """Fixes for huss."""
    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return cubes


class SfcWind(Fix):
    """Fixes for sfcWind."""
    def fix_metadata(self, cubes):
        """Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


class Tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return cubes


class Uas(Fix):
    """Fixes for uas."""
    def fix_metadata(self, cubes):
        """Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes


class Vas(Fix):
    """Fixes for vas."""
    def fix_metadata(self, cubes):
        """Add height (10m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 10.0)
        return cubes
