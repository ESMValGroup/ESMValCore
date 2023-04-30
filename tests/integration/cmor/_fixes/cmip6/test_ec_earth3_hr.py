"""Tests for EC-Earth3-HR."""
from esmvalcore.cmor._fixes.cmip6.ec_earth3_hr import Ofx, Omon
from esmvalcore.cmor._fixes.common import NemoGridFix
from esmvalcore.cmor.fix import Fix


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'EC-Earth3-HR', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'EC-Earth3-HR', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
