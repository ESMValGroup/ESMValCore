"""Tests for the fixes of CORDEX."""

from esmvalcore.cmor._fixes.cordex.project import AllVars
from esmvalcore.cmor.fix import Fix


def test_get_allvars_fix():
    fix = Fix.get_fixes('CORDEX', 'any_dataset', 'mip', 'tas')
    assert fix == [AllVars(None)]
