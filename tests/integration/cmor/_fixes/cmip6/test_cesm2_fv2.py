"""Tests for the fixes of CESM2-FV2."""
from esmvalcore.cmor._fixes.cmip6.cesm2 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2_fv2 import Cl, Cli, Clw, Tas
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is BaseCl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is BaseCl


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2-FV2', 'Amon', 'tas')
    assert fix == [Tas(None)]


def test_tas_fix():
    """Test fix for ``tas``."""
    assert Tas is BaseTas
