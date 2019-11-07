"""Fixes for CESM2-WACCM model."""
from .cesm2 import Tas as BaseTas
from .cesm2 import msftmz as BaseMsftmz



class Tas(BaseTas):
    """Fixes for tas."""


class msftmz(BaseMsftmz):
    """Fix msftmz."""
