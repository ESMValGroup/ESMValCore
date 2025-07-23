"""Fixes for NorESM2-MM model."""

from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg_lr import (
    AllVars as BaseAllVars,
)
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord

AllVars = BaseAllVars

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord
