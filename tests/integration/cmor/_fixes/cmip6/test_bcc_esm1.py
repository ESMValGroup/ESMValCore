"""Test fixes for BCC-ESM1."""
from esmvalcore.cmor._fixes.cmip6.bcc_esm1 import (
    Cl,
    Cli,
    Clw,
    Siconc,
    Sos,
    Tos,
)
from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is OceanFixGrid


def test_get_sos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Omon', 'sos')
    assert fix == [Sos(None)]


def test_sos_fix():
    """Test fix for ``sos``."""
    assert Sos is OceanFixGrid


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Omon', 'tos')
    assert fix == [Tos(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid
