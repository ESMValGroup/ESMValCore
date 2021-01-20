"""Fixes for CESM2-WACCM-FV2 model."""
from .cesm2 import Tas as BaseTas
from .cesm2_waccm import Cl as BaseCl, Cli as BaseCli, Clw as BaseClw


Cl = BaseCl


Cli = BaseCli


Clw = BaseClw


Tas = BaseTas
