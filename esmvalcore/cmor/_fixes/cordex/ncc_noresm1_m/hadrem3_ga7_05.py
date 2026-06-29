"""Fixes for rcm HadREM3-GA7-05 driven by NCC-NorESM1-M."""

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5.hadrem3_ga7_05 import (
    Sftlf as BaseSftlf,
)
from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    MOHCHadREM3GA705 as BaseFix,
)

AllVars = BaseFix


class Sftlf(BaseSftlf):
    """Fixes for sftlf."""
