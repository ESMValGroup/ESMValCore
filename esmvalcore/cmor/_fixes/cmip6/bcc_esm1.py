"""Fixes for BCC-ESM1 model."""
from ..common import ClFixHybridPressureCoord
from .bcc_csm2_mr import Tos as BaseTos
from .bcc_csm2_mr import Thetao as BaseThetao

Cl = ClFixHybridPressureCoord

Cli = ClFixHybridPressureCoord

Clw = ClFixHybridPressureCoord

Tos = BaseTos

Thetao = BaseThetao

So = BaseThetao
