"""Fixes for bcc-csm1-1."""

from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)

Cl = ClFixHybridPressureCoord


Tos = OceanFixGrid
