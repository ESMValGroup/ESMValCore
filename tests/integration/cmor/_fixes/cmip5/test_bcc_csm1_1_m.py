"""Test fixes for bcc-csm1-1-m."""

from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1_m import Cl, Tos
from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)
from esmvalcore.cmor._fixes.fix import Fix, GenericFix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'bcc-csm1-1-m', 'Amon', 'cl')
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'bcc-csm1-1-m', 'Omon', 'tos')
    assert fix == [Tos(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid
