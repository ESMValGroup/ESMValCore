"""Tests for the fixes of CMCC-CM2-HR4."""

from esmvalcore.cmor._fixes.cmip6.cmcc_cm2_hr4 import Ofx, Omon
from esmvalcore.cmor._fixes.common import NemoGridFix
from esmvalcore.cmor._fixes.fix import Fix


def test_get_omon_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'CMCC-CM2-HR4', 'Omon', 'tos')
    assert Omon(None) in fixes


def test_get_ofx_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'CMCC-CM2-HR4', 'Ofx', 'areacello')
    assert Ofx(None) in fixes


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
