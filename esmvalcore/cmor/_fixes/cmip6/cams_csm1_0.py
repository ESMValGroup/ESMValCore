"""Fixes for CAMS-CSM1-0 model."""
from ..common import (ClFixHybridPressureCoord, CliFixHybridPressureCoord,
                      ClwFixHybridPressureCoord)

Cl = ClFixHybridPressureCoord


Cli = CliFixHybridPressureCoord


Clw = ClwFixHybridPressureCoord
