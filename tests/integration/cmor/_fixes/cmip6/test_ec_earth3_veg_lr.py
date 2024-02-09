"""Test fixes for EC-Earth3-Veg-LR."""
from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg_lr import Siconc
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import Fix, GenericFix


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg-LR', 'SImon', 'siconc')
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is OceanFixGrid
