"""Fixes for BCC-ESM1 model."""

from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Tos = OceanFixGrid


Sos = OceanFixGrid


So = OceanFixGrid


Siconc = OceanFixGrid
