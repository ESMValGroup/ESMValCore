"""Fixes for CESM2-FV2 model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Tas as BaseTas
from .cesm2 import Omon as baseOmon

from ..common import SiconcFixScalarCoord


Cl = BaseCl


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2
no3 = BaseFgco2
#Omon = baseOmon

Siconc = SiconcFixScalarCoord


Tas = BaseTas
