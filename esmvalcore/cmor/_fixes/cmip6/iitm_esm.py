"""Fixes for IITM-ESM model."""

import logging

import numpy as np

from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.fixes import get_time_bounds

logger = logging.getLogger(__name__)

Tos = OceanFixGrid


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        IITM-ESM monthly data may have a bad time bounds spanning 20 days.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            freq = self.extra_facets["frequency"]
            time = cube.coord("time", dim_coords=True)
            bounds = get_time_bounds(time, freq)
            if np.any(bounds != time.bounds):
                time.bounds = bounds
        logger.warning(
            "Using 'area_weighted' regridder scheme in Omon variables "
            "for dataset %s causes discontinuities in the longitude "
            "coordinate.",
            self.extra_facets["dataset"],
        )
        return cubes
