"""Fixes for CESM2-WACCM-FV2 model."""

import iris
import numpy as np

from ..common import SiconcFixScalarCoord
from ..fix import Fix
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Tas as BaseTas
from .cesm2 import Pr as BasePr
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw

Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Tas = BaseTas


Pr = BasePr
