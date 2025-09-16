"""Test fixes for GISS-E2-1-G-CC."""

from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g import Tos as BaseTos
from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g_cc import Cl, Cli, Clw, Tos
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix, GenericFix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GISS-E2-1-G-CC", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GISS-E2-1-G-CC", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "GISS-E2-1-G-CC", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is BaseTos
