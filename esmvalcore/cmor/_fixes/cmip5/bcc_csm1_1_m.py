"""Fixes for bcc-csm1-1-m."""
from .bcc_csm1_1 import Tos as BaseTos
from ..common import ClFixHybridPressureCoord


Cl = ClFixHybridPressureCoord


class Tos(BaseTos):
    """Fixes for tos."""
