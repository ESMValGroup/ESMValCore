"""Fixes for rcm CNRM-ALADIN63 driven by CNRM-CERFACS-CNRM-CM5."""
from esmvalcore.cmor._fixes.cordex.cordex_fixes import TimeLongName as BaseFix
from esmvalcore.cmor._fixes.cordex.cordex_fixes import LambertGrid as GridFix


Pr = BaseFix

AllVars = GridFix
