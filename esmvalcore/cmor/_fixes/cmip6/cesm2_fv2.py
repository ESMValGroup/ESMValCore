"""Fixes for CESM2-FV2 model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas
from ..common import SiconcFixScalarCoord
from ..fix import Fix


Cl = BaseCl


Cli = Cl


Clw = Cl


Siconc = SiconcFixScalarCoord


Tas = BaseTas
