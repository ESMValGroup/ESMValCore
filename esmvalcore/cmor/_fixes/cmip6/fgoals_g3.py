"""Fixes for FGOALS-g3 model."""
from ..cmip5.fgoals_g2 import Cl as BaseCl
from ..common import OceanFixGrid


from ..common import ClFixmsftmzbasin, ClFixmsftyzbasin

msftmz = ClFixmsftmzbasin
msftyz = ClFixmsftyzbasin



Cl = BaseCl


Cli = BaseCl


Clw = BaseCl


Tos = OceanFixGrid


Siconc = OceanFixGrid
