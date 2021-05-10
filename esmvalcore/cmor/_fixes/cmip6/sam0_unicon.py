"""Fixes for SAM0-UNICON model."""
from ..common import ClFixHybridPressureCoord
from ..giss_e2_1_g import Nbp as BaseNbp

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Nbp = BaseNbp
