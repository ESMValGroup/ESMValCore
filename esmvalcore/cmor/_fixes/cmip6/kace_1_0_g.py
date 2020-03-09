"""Fixes for KACE-1-0-G."""
from .ukesm1_0_ll import Cl as BaseCl


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw``."""


class Cli(Cl):
    """Fixes for ``cli``."""
