"""Fixes for ERA5-Land."""
import logging

from esmvalcore.cmor._fixes.native6.era5 import (Pr,
                                                 AllVars)


logger = logging.getLogger(__name__)
logger.info("Load classes from era5.py")
logger.info(Pr)
logger.info(AllVars)
