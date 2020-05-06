"""Tests for the fixes of FGOALS-g3."""
from esmvalcore.cmor._fixes.cmip5.fgoals_g2 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.fgoals_g3 import Cl, Cli, Clw
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FGOALS-g3', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FGOALS-g3', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is BaseCl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FGOALS-g3', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is BaseCl
