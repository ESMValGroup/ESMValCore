"""Fixes for CanESM5-CanOE model."""

from .canesm5 import Co2 as BaseCO2
from .canesm5 import Gpp as BaseGpp


class Co2(BaseCO2):
    """Fixes for co2."""


class Gpp(BaseGpp):
    """Fixes for gpp."""
