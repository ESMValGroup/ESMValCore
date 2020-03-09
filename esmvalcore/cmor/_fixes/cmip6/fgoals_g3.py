"""Fixes for FGOALS-g3 model."""
from ..cmip5.fgoals_g2 import Cl as BaseCl


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw``."""


class Cli(Cl):
    """Fixes for ``cli``."""
