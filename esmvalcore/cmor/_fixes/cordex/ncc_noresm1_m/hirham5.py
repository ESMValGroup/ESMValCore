"""Fixes for rcm HIRHAM5 driven by ICHEC-EC-EARTH."""

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5.hirham5 import (
    Clivi as BaseClivi,
)


class Clivi(BaseClivi):
    """Fixes for variable clivi."""
