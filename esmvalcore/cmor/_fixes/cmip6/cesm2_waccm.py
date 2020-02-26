"""Fixes for CESM2-WACCM model."""
from .cesm2 import Tas as BaseTas


class Tas(BaseTas):
    """Fixes for tas."""
