"""Tests for EC-Earth3-Veg."""

from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg import Ofx, Omon
from esmvalcore.cmor._fixes.common import NemoGridFix
from esmvalcore.cmor.fix import Fix


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-Veg', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-Veg', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
