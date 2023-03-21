"""Fixes for rcm CNRM-ALADIN63 driven by CNRM-CERFACS-CNRM-CM5."""
import iris
import numpy as np

from esmvalcore.cmor._fixes.cordex.cordex_fixes import TimeLongName as BaseFix
from esmvalcore.cmor._fixes.cordex.cordex_fixes import LambertGrid as GridFix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord


Pr = BaseFix

AllVars = GridFix
