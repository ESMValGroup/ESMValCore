"""Test fixes for EC-Earth3-Veg-LR."""
from esmvalcore.cmor._fixes.common import NemoGridFix
from esmvalcore.cmor._fixes.ecearth.ec_earth3_veg_lr import Ofx, Omon
from esmvalcore.cmor._fixes.fix import Fix


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-Veg-LR', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-Veg-LR', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
