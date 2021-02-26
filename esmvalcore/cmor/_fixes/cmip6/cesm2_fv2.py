"""Fixes for CESM2-FV2 model."""
from .cesm2 import Cl as BaseCl
from .cesm2 import Tas as BaseTas
from .gfdl_esm4 import Siconc as Addtypesi


Cl = BaseCl


Cli = Cl


Clw = Cl


Tas = BaseTas


Siconc = Addtypesi
