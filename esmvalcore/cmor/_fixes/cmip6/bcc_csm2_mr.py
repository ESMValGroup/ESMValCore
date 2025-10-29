"""Fixes for BCC-CSM2-MR model."""

from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)

Areacello = OceanFixGrid


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Tos = OceanFixGrid


Siconc = OceanFixGrid


Sos = OceanFixGrid


Uo = OceanFixGrid
