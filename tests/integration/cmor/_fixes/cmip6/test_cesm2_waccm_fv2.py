"""Tests for the fixes of CESM2-WACCM-FV2."""
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm_fv2 import Cl, Cli, Clw
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM-FV2', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM-FV2', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM-FV2', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl
