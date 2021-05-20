"""Tests for the fixes of KACE-1-0-G."""
from esmvalcore.cmor._fixes.cmip6.kace_1_0_g import Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridHeightCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'KACE-1-0-G', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridHeightCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'KACE-1-0-G', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridHeightCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'KACE-1-0-G', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridHeightCoord
