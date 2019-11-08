"""Fixes for MPI-ESM1-2-XR model."""

from .mpi_esm1_2_hr import Tas as BaseTas
from .mpi_esm1_2_hr import Ta as BaseTa
from .mpi_esm1_2_hr import SfcWind as BaseSfcWind


class Tas(BaseTas):
    """Fixes for tas."""


class Ta(BaseTa):
    """Fixes for ta."""

class Va(BaseTa):
    """Fixes for va."""


class Zg(BaseTa):
    """Fixes for zg."""


class Ua(BaseTa):
    """Fixes for ua."""


class SfcWind(BaseSfcWind):
    """Fixes for sfcWind."""
