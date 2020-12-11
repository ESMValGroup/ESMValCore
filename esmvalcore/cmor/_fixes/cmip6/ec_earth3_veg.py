"""Fixes for EC-Earth3-Veg model."""
import numpy as np

from ..fix import Fix


class Siconca(Fix):
    """Fixes for siconca."""
    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube which needs to be fixed.

        Returns
        -------
        iris.cube.Cube
        """
        cube.data = cube.core_data() * 100.
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix latitude points and bounds.

        Fix latitude_bounds and longitude_bounds data type and round to 8 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)

        for cube in cubes:
            latitude = cube.coord('latitude')
            latitude.points = np.round(latitude.points, 8)
            latitude.bounds = np.round(latitude.bounds, 8)

        return cubes
