"""Tests for EC-Earth3-AerChem."""
from esmvalcore.cmor._fixes.common import NemoGridFix
from esmvalcore.cmor._fixes.ecearth.ec_earth3_aerchem import Ofx, Omon
from esmvalcore.cmor.fix import Fix


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-AerChem', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('ECEARTH', 'EC-Earth3-AerChem', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_ofx_fix():
    """Test fix for ``Omon``."""
    assert Ofx is NemoGridFix
