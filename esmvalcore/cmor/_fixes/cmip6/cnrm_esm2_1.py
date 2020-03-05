"""Fixes for CNRM-ESM2-1 model."""
from .cnrm_cm6_1 import Cl as BaseCl, Clcalipso as BaseClcalipso


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(Cl):
    """Fixes for ``cli (same as for cl)``."""


class Clcalipso(BaseClcalipso):
    """Fixes for ``clcalipso``."""
