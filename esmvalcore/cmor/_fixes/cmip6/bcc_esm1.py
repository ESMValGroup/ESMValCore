"""Fixes for BCC-ESM1 model."""
from ..common import ClFixHybridPressureCoord
from .bcc_csm2_mr import Tos as BaseTos


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Tos = BaseTos
