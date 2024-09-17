"""Fixes for MPI-ESM1-2-XR model."""

from .mpi_esm1_2_hr import Tas as BaseTas
from .mpi_esm1_2_hr import Ta as BaseFix
from .mpi_esm1_2_hr import SfcWind as BaseSfcWind


class Tas(BaseTas):
    """Fixes for tas."""


class Tasmax(BaseTas):
    """Fixes for tasmax."""


class Tasmin(BaseTas):
    """Fixes for tasmin."""


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


class SfcWindmax(BaseSfcWind):
    """Fixes for sfcWindmax."""


class Uas(BaseSfcWind):
    """Fixes for uas."""


class Vas(BaseSfcWind):
    """Fixes for vas."""
