"""Fixes for ERA5-Land."""

from loguru import logger

from esmvalcore.cmor._fixes.native6.era5 import (
    AllVars,
    Evspsbl,
    Evspsblpot,
    Pr,
)

logger.info("Load classes from era5.py")
logger.info(Pr)
logger.info(Evspsbl)
logger.info(Evspsblpot)
logger.info(AllVars)
