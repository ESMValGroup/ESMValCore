"""Fixes for FIO-ESM-2-0 model."""
from ..common import OceanFixGrid
from ..fix import Fix
from ..shared import round_coordinates

Tos = OceanFixGrid

class Omon(Fix):
    """Fixes for Omon."""

    def fix_metadata(self, cubes):
        """
        Fix latitude and longitude rounding to 6 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        round_coordinates(cubes, decimals=6, coord_names=["longitude", "latitude"])
        return cubes