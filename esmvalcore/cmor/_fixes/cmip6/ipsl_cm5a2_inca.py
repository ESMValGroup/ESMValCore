"""Fixes for IPSL-CM5A2-INCA model."""
from .ipsl_cm6a_lr import AllVars as BaseAllVars
from .ipsl_cm6a_lr import Clcalipso as BaseClcalipso
from .ipsl_cm6a_lr import Omon as BaseOmon

from ..common import OceanFixGrid
from ..shared import fix_ocean_depth_coord


AllVars = BaseAllVars


Clcalipso = BaseClcalipso


Omon = BaseOmon

