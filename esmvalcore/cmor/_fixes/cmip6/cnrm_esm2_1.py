"""Fixes for CNRM-ESM2-1 model."""
from .cnrm_cm6_1 import Cl as BaseCl
from .cnrm_cm6_1 import Clcalipso as BaseClcalipso


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clcalipso(BaseClcalipso):
    """Fixes for ``clcalipso``."""


class Cli(Cl):
    """Fixes for ``cli``."""


class Clw(Cl):
    """Fixes for ``clw``."""
