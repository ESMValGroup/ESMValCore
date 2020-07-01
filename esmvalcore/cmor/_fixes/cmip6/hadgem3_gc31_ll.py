"""Fixes for CMIP6 HadGEM-GC31-LL."""
from ..common import ClFixHybridHeightCoord
from .ukesm1_0_ll import AllVars as BaseAllVars


AllVars = BaseAllVars


Cl = ClFixHybridHeightCoord


Cli = ClFixHybridHeightCoord


Clw = ClFixHybridHeightCoord
