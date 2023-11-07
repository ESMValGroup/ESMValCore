"""Fixes for KACE-1-0-G."""
import logging

import numpy as np

from esmvalcore.cmor.fixes import get_time_bounds

from ..common import ClFixHybridHeightCoord, OceanFixGrid
from ..fix import Fix

logger = logging.getLogger(__name__)

Cl = ClFixHybridHeightCoord

Cli = ClFixHybridHeightCoord

Clw = ClFixHybridHeightCoord

Tos = OceanFixGrid


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        KACE-1-0-G mon data may have a bad time bounds spanning 20 days.

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
            self.extra_facets['dataset'],
        )
        return cubes
