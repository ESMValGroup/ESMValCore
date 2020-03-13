"""Fixes for MIROC6 model."""
from ..common import (ClFixHybridPressureCoord, CliFixHybridPressureCoord,
                      ClwFixHybridPressureCoord)

Cl = ClFixHybridPressureCoord


Cli = CliFixHybridPressureCoord


Clw = ClwFixHybridPressureCoord
