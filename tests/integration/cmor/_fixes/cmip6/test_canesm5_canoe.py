"""Test fixes for CanESM5-CanOE."""
from esmvalcore.cmor._fixes.cmip6.canesm5 import Co2 as BaseCo2
from esmvalcore.cmor._fixes.cmip6.canesm5 import Gpp as BaseGpp
from esmvalcore.cmor._fixes.cmip6.canesm5_canoe import Co2, Gpp
from esmvalcore.cmor._fixes.fix import Fix


def test_get_co2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CanESM5-CanOE', 'Amon', 'co2')
    assert fix == [Co2(None)]


def test_co2_fix():
    """Test fix for ``co2``."""
    assert issubclass(Co2, BaseCo2)


def test_get_gpp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CanESM5-CanOE', 'Lmon', 'gpp')
    assert fix == [Gpp(None)]


def test_gpp_fix():
    """Test fix for ``gpp``."""
    assert issubclass(Gpp, BaseGpp)
