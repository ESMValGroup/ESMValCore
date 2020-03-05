"""Fixes for CMIP6 ACCESS-ESM1-5."""
from .ukesm1_0_ll import Cl as BaseCl


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(BaseCl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(BaseCl):
    """Fixes for ``cli (same as for cl)``."""
