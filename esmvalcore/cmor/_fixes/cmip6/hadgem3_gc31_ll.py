"""Fixes for CMIP6 HadGEM-GC31-LL."""
from ..common import (ClFixHybridHeightCoord, CliFixHybridHeightCoord,
                      ClwFixHybridHeightCoord)
from .ukesm1_0_ll import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all vars."""


Cl = ClFixHybridHeightCoord


Cli = CliFixHybridHeightCoord


Clw = ClwFixHybridHeightCoord
