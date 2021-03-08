"""Fixes for BCC-ESM1 model."""
from ..common import ClFixHybridPressureCoord
from .bcc_csm2_mr import Tos as BaseTos, so, thetao


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord

so = so
thetao = thetao
Tos = BaseTos
