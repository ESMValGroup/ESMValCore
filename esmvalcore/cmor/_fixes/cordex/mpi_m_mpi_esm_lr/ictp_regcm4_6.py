"""Fixes for rcm ICTP-RegCM4-6 driven by MPI-M-MPI-ESM-LR."""
from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    TimeLongName as BaseFix)

from esmvalcore.cmor._fixes.cordex.cordex_fixes import LambertGrid as GridFix

Pr = BaseFix

Tas = BaseFix

AllVars = GridFix
