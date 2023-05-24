"""Fixes for FIO-ESM-2-0 model."""
import iris

from ..common import OceanFixGrid
from ..fix import Fix
from ..shared import round_coordinates

Tos = OceanFixGrid


class Omon(Fix):
    """Fixes for Omon vars."""

    def fix_metadata(self, cubes):
        """Fix latitude and longitude with round to 6 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        round_coordinates(cubes,
                          decimals=6,
                          coord_names=["longitude", "latitude"])
        return cubes


class Amon(Fix):
    """Fixes for Amon vars."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        FIO-ESM-2-0 Amon data contains error in coordinate bounds.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            # Check both lat and lon coords and replace bounds if necessary
            latitude = cube.coord("latitude")
            if latitude.has_bounds():
                if latitude.bounds[1:, 0] != latitude.bounds[:-1, 1]:
                    latitude.bounds = None
                    latitude.guess_bounds()
                    iris.util.promote_aux_coord_to_dim_coord(cube, "latitude")

            longitude = cube.coord("longitude")
            if longitude.has_bounds():
                if longitude.bounds[1:, 0] != longitude.bounds[:-1, 1]:
                    longitude.bounds = None
                    longitude.guess_bounds()
                    iris.util.promote_aux_coord_to_dim_coord(cube, "longitude")
        return cubes
