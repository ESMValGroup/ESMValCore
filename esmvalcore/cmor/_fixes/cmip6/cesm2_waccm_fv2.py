"""Fixes for CESM2-WACCM-FV2 model."""
from .cesm2 import Tas as BaseTas
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw

Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


Tas = BaseTas

Fgco2 = BaseFgco2
