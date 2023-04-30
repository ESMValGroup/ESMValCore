"""Test fixes for CNRM-CM6-1-HR."""
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Cli as BaseCli
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import Clw as BaseClw
from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1_hr import Cl, Cli, Clw, Ofx, Omon
from esmvalcore.cmor._fixes.common import NemoGridFix
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


def test_get_omon_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1-HR', 'Omon', 'thetao')
    assert fix == [Omon(None)]


def test_omon_fix():
    """Test fix for ``Omon``."""
    assert Omon is NemoGridFix


def test_get_ofx_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1-HR', 'Ofx', 'areacello')
    assert fix == [Ofx(None)]


def test_ofx_fix():
    """Test fix for ``Ofx``."""
    assert Ofx is NemoGridFix
