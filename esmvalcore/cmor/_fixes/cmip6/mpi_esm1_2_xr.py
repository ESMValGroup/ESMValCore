"""Fixes for MPI-ESM1-2-XR model."""

from .mpi_esm1_2_hr import Tas as BaseTas
from .mpi_esm1_2_hr import Ta as BaseFix
from .mpi_esm1_2_hr import SfcWind as BaseSfcWind


class Tas(BaseTas):
    """Fixes for tas."""


class Ta(BaseFix):
    """Fixes for ta."""


class Va(BaseFix):
    """Fixes for va."""


class Zg(BaseFix):
    """Fixes for zg."""


class Ua(BaseFix):
    """Fixes for ua."""


class SfcWind(BaseSfcWind):
    """Fixes for sfcWind."""
