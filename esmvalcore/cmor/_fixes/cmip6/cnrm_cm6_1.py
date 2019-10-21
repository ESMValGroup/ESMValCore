"""Fixes for CNRM-CM6-1 model."""
from .cnrm_esm2_1 import Abs550aer as BaseAbs550aer
from .cnrm_esm2_1 import Od550aer as BaseOd550aer

class Abs550aer(BaseAbs550aer):
    """Fixes for abs550aer"""
class Od550aer(BaseOd550aer):
    """Fixes for od550aer"""
