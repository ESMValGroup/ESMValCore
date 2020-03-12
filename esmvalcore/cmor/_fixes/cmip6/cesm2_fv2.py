"""Fixes for CESM2-FV2 model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw``."""


class Cli(Cl):
    """Fixes for ``cli``."""


class Tas(BaseTas):
    """Fixes for ``tas``."""
