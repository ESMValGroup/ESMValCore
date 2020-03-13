"""Fixes for MRI-ESM2-0 model."""
from ..common import (ClFixHybridPressureCoord, CliFixHybridPressureCoord,
                      ClwFixHybridPressureCoord)

Cl = ClFixHybridPressureCoord


Cli = CliFixHybridPressureCoord


Clw = ClwFixHybridPressureCoord
