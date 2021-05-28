"""Test fixes for CSIRO-Mk3-6-0."""
from esmvalcore.cmor._fixes.cmip5.csiro_mk3_6_0 import Cl
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'CSIRO-Mk3-6-0', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord
