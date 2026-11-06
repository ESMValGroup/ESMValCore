"""Test fixes for BCC-CSM2-MR."""

from esmvalcore.cmor._fixes.cmip6.bcc_csm2_mr import (
    Areacello,
    Siconc,
    Sos,
    Tos,
)
from esmvalcore.cmor._fixes.common import (
    OceanFixGrid,
)
from esmvalcore.cmor._fixes.fix import Fix, GenericFix


def test_get_areacello_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "Amon", "areacello")
    assert fix == [Areacello(None), GenericFix(None)]


def test_areacello_fix():
    """Test fix for ``areacello``."""
    assert Areacello is OceanFixGrid


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "Omon", "tos")
    assert fix == [Tos(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is OceanFixGrid


def test_get_sos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "Omon", "sos")
    assert fix == [Sos(None), GenericFix(None)]


def test_sos_fix():
    """Test fix for ``sos``."""
    assert Sos is OceanFixGrid
