"""Tests for the fixes of E3SM-1-1."""

import numpy as np
import pytest
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip6.e3sm_1_1 import Hus, Ta, Ua, Va
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from tests import assert_array_equal


def test_get_hus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "E3SM-1-1", "Amon", "hus")
    assert fix == [Hus(None), GenericFix(None)]


def test_get_ta_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "E3SM-1-1", "Amon", "ta")
    assert fix == [Ta(None), GenericFix(None)]


def test_get_ua_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "E3SM-1-1", "Amon", "ua")
    assert fix == [Ua(None), GenericFix(None)]


def test_get_va_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "E3SM-1-1", "Amon", "va")
    assert fix == [Va(None), GenericFix(None)]


@pytest.mark.parametrize("lazy", [True, False])
def test_hus_fix(lazy):
    """Test fix for ``hus``."""
    cube = Cube([1.0, 1e35])
    if lazy:
        cube.data = cube.lazy_data()

    fix = Hus(None)

    fixed_cube = fix.fix_data(cube)

    assert fixed_cube is cube
    assert fixed_cube.has_lazy_data() is lazy
    assert_array_equal(fixed_cube.data, np.ma.masked_invalid([1.0, np.nan]))


def test_ta_fix():
    """Test fix for ``ta``."""
    assert Ta == Hus


def test_ua_fix():
    """Test fix for ``ua``."""
    assert Ua == Hus


def test_va_fix():
    """Test fix for ``va``."""
    assert Va == Hus
