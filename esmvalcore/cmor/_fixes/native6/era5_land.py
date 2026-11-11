"""Fixes for ERA5-Land."""

import logging

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.native6.era5 import (
    AllVars,
    Evspsbl,
    Evspsblpot,
    Pr,
)

logger = logging.getLogger(__name__)
logger.info("Load classes from era5.py")
logger.info(Pr)
logger.info(Evspsbl)
logger.info(Evspsblpot)
logger.info(AllVars)


class Snw(Fix):
    """Fixes for snw, derived from the ERA5-Land variable ``sd``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        for cube in cubes:
            # Invalid input cube units are 'm of water equivalent'
            # Assumption: 1 m of water equivalent corresponds to 1000 kg m-2.
            cube.data = cube.core_data() * 1000.0
            cube.units = "kg m-2"
        return cubes
