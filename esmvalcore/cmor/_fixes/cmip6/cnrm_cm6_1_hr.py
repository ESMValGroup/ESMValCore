"""Fixes for CNRM-CM6-1-HR model."""
from .cnrm_cm6_1 import Cl as BaseCl
from .cnrm_cm6_1 import Cli as BaseCli
from .cnrm_cm6_1 import Clw as BaseClw
from ..common import NemoGridFix


Cl = BaseCl


Cli = BaseCli


Clw = BaseClw

Omon = NemoGridFix
