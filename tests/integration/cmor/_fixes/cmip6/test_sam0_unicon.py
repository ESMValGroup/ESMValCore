"""Test fixes for SAM0-UNICON."""
from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g import Nbp as BaseNbp
from esmvalcore.cmor._fixes.cmip6.sam0_unicon import Cl, Cli, Clw, Nbp
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_get_nbp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'SAM0-UNICON', 'Lmon', 'nbp')
    assert fix == [Nbp(None)]


def test_nbp_fix():
    """Test fix for ``nbp``."""
    assert Nbp is BaseNbp
