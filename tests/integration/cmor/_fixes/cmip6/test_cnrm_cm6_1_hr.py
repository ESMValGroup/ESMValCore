"""Test fixes for CNRM-CM6-1-HR."""
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cli as BaseCli
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Clw as BaseClw
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1_hr import Cl, Cli, Clw
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1-HR', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1-HR', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is BaseCli


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1-HR', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is BaseClw
