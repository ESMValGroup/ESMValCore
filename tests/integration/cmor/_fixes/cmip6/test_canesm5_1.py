"""Tests for the fixes of CanESM5-1-1."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.canesm5_1 import Cl, Cli, Clw, Co2, Gpp, Ps
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


def test_get_co2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Amon", "co2")
    assert fix == [Co2(None), GenericFix(None)]


@pytest.fixture
def co2_cube():
    """``co2`` cube."""
    return iris.cube.Cube(
        [1.0],
        var_name="co2",
        standard_name="mole_fraction_of_carbon_dioxide_in_air",
        units="mol mol-1",
    )


def test_co2_fix_data(co2_cube):
    """Test ``fix_data`` for ``co2``."""
    fix = Co2(None)
    out_cube = fix.fix_data(co2_cube)
    np.testing.assert_allclose(out_cube.data, [1.0e-6])


@pytest.fixture
def gpp_cube():
    """``gpp`` cube."""
    return iris.cube.Cube(
        [0, 1],
        var_name="gpp",
        standard_name="gross_primary_productivity_of_biomass_expressed_as_"
        "carbon",
        units="kg m-2 s-1",
    )


def test_get_gpp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Lmon", "gpp")
    assert fix == [Gpp(None), GenericFix(None)]


def test_gpp_fix_data(gpp_cube):
    """Test ``fix_data`` for ``gpp``."""
    fix = Gpp(None)
    out_cube = fix.fix_data(gpp_cube)
    np.testing.assert_allclose(
        out_cube.data,
        np.ma.masked_invalid([np.nan, 1]),
    )
    assert np.array_equal(out_cube.data.mask, [True, False])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, ClFixHybridPressureCoord)


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert issubclass(Cli, ClFixHybridPressureCoord)


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert issubclass(Clw, ClFixHybridPressureCoord)


def test_get_ps_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5-1", "Amon", "ps")
    assert fix == [Ps(None), GenericFix(None)]


def test_ps_fix():
    """Test fix for ``ps``."""
    assert issubclass(Ps, ClFixHybridPressureCoord)
