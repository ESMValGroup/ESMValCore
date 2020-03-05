"""Fixes for CMIP6 KACE-1-0-G."""
from .ukesm1_0_ll import Cl as BaseCl
from .ukesm1_0_ll import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all vars."""


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(BaseCl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(BaseCl):
    """Fixes for ``cli (same as for cl)``."""
