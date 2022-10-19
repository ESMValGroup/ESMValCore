"""Fixes for rcm CLMcom-CCLM4-8-17 driven by MIROC-MIROC5."""
from esmvalcore.cmor.fix import Fix

from cf_units import Unit
import numpy as np


from ..cordex_fixes import CLMcomCCLM4817 as BaseFix

AllVars = BaseFix
