"""Test fixes for CNRM-ESM2-1."""
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Clcalipso as BaseClcalipso
from esmvalcore.cmor._fixes.cmip6.cnrm_esm2_1 import Cl, Clcalipso, Cli, Clw
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl(None) == BaseCl(None)


def test_get_clcalipso_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clcalipso')
    assert fix == [Clcalipso(None)]


def test_clcalipso_fix():
    """Test fix for ``cl``."""
    assert Clcalipso(None) == BaseClcalipso(None)


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli(None) == BaseCl(None)


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-ESM2-1', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw(None) == BaseCl(None)
