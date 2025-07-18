"""Fixes for CESM2-FV2 model."""

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord

from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Pr as BasePr
from .cesm2 import Tas as BaseTas

Cl = BaseCl


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Tas = BaseTas


Pr = BasePr
