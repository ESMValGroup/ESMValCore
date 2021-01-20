"""Tests for the fixes of CAS-ESM2-0."""
from esmvalcore.cmor._fixes.cmip6.cas_esm2_0 import Cl
from esmvalcore.cmor._fixes.cmip6.ciesm import Cl as BaseCl
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CAS-ESM2-0', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl
