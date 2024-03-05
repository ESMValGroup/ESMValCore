"""Fixes for FIO-ESM-2-0 model."""
import logging

import numpy as np

from ..common import OceanFixGrid
from ..fix import Fix
from ..shared import round_coordinates

logger = logging.getLogger(__name__)

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
        logger.warning(
            "Using 'area_weighted' regridder scheme in Omon variables "
            "for dataset %s causes discontinuities in the longitude "
            "coordinate.",
            self.extra_facets['dataset'],
        )
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
                if np.any(latitude.bounds[1:, 0] != latitude.bounds[:-1, 1]):
                    latitude.bounds = None
                    latitude.guess_bounds()

            longitude = cube.coord("longitude")
            if longitude.has_bounds():
                if np.any(longitude.bounds[1:, 0] != longitude.bounds[:-1, 1]):
                    longitude.bounds = None
                    longitude.guess_bounds()
        return cubes


class Clt(Fix):
    """Fixes for clt."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube
        """
        if cube.core_data().max() <= 1.0:
            cube.units = '1'
            cube.convert_units('%')
        return cube
