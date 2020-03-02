"""Fixes for CESM2-WACCM model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas


class Cl(BaseCl):
    """Fixes for cl."""


class Tas(BaseTas):
    """Fixes for tas."""
