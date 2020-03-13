"""Fixes for BCC-ESM1 model."""
from ..common import (ClFixHybridPressureCoord, CliFixHybridPressureCoord,
                      ClwFixHybridPressureCoord)
from .bcc_csm2_mr import Tos as BaseTos

Cl = ClFixHybridPressureCoord


Cli = CliFixHybridPressureCoord


Clw = ClwFixHybridPressureCoord


class Tos(BaseTos):
    """Fixes for tos."""
