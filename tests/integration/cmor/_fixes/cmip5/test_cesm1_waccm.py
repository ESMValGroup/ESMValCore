"""Tests for CESM1-WACCM fixes."""
from esmvalcore.cmor._fixes.cmip5.cesm1_cam5 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.cesm1_waccm import Cl
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'CESM1-WACCM', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl
