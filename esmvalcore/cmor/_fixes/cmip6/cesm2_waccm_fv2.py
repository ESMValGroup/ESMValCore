"""Fixes for CESM2-WACCM-FV2 model."""
from .cesm2 import Tas as BaseTas
from .cesm2_waccm import Cl as BaseCl
from .cesm2_waccm import Cli as BaseCli
from .cesm2_waccm import Clw as BaseClw
from .gfdl_esm4 import Siconc as Addtypesi

Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


Tas = BaseTas


Siconc = Addtypesi
