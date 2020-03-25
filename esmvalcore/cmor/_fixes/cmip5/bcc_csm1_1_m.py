"""Fixes for bcc-csm1-1-m."""
from ..common import ClFixHybridPressureCoord
from .bcc_csm1_1 import Tos as BaseTos


Cl = ClFixHybridPressureCoord


Tos = BaseTos
