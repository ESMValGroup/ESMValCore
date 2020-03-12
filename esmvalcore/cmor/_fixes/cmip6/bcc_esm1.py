"""Fixes for BCC-ESM1 model."""
from .bcc_csm2_mr import Tos as BaseTos
from ..common import ClFixHybridPressureCoord


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


class Tos(BaseTos):
    """Fixes for tos."""
