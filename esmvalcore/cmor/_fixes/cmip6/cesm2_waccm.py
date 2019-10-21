"""Fixes for CESM2-WACCM model."""
from .cesm2 import Tas as BaseTas
from .cesm2 import Abs550aer as BaseAbs550aer
from .cesm2 import Od550aer as BaseOd550aer
from ..fix import Fix
from ..shared import add_scalar_height_coord
import iris



class Tas(BaseTas):
    """Fixes for tas."""

class Abs550aer(BaseAbs550aer):
    """Fixes for abs550aer"""
class Od550aer(BaseOd550aer):
    """Fixes for od550aer"""
