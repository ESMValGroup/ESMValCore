"""Fixes for GISS-E2-1-G-CC model."""

from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord

from .giss_e2_1_g import Tos as BaseTos

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Tos = BaseTos
