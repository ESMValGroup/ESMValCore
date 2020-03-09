"""Fixes for GISS-E2-1-G model."""
from ..cmip5.bcc_csm1_1 import Cl as BaseCl


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Cli(Cl):
    """Fixes for ``cli``."""


class Clw(Cl):
    """Fixes for ``clw``."""
