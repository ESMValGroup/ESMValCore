"""Fixes for rcm HIRHAM5 driven by IPSL-IPSL-CM5A-MR."""

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5.hirham5 import (
    Clivi as BaseClivi,
)


class Clivi(BaseClivi):
    """Fixes for variable clivi."""
