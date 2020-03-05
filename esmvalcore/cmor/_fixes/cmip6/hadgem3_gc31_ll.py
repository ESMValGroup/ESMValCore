"""Fixes for CMIP6 HadGEM-GC31-LL."""
from .ukesm1_0_ll import Cl as BaseCl, AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all vars."""


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(Cl):
    """Fixes for ``cli (same as for cl)``."""
