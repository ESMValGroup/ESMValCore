"""Test fixes for EC-Earth3-Veg-LR."""
from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg_lr import Ofx, Omon, Siconc
from esmvalcore.cmor._fixes.common import NemoGridFix, OceanFixGrid
from esmvalcore.cmor._fixes.fix import Fix


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg-LR', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is OceanFixGrid


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg-LR', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg-LR', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
