"""Tests for the fixes of TaiESM1."""
from esmvalcore.cmor._fixes.cmip6.taiesm1 import Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'TaiESM1', 'Amon', 'cl')
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'TaiESM1', 'Amon', 'cli')
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'TaiESM1', 'Amon', 'clw')
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord
