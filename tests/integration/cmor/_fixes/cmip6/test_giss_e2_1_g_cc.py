"""Test fixes for GISS-E2-1-G-CC."""

from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g import Tos as BaseTos
from esmvalcore.cmor._fixes.cmip6.giss_e2_1_g_cc import Tos


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is BaseTos
