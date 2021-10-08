"""Tests for the fixes of CanESM5."""
import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.canesm5 import Co2, Gpp
from esmvalcore.cmor.fix import Fix


def test_get_co2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CanESM5', 'Amon', 'co2')
    assert fix == [Co2(None)]


@pytest.fixture
def co2_cube():
    """``co2`` cube."""
    cube = iris.cube.Cube(
        [1.0],
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='mol mol-1',
    )
    return cube


def test_co2_fix_data(co2_cube):
    """Test ``fix_data`` for ``co2``."""
    fix = Co2(None)
    out_cube = fix.fix_data(co2_cube)
    np.testing.assert_allclose(out_cube.data, [1.e-6])


@pytest.fixture
def gpp_cube():
    """``gpp`` cube."""
    cube = iris.cube.Cube(
        [0, 1],
        var_name='gpp',
        standard_name='gross_primary_productivity_of_biomass_expressed_as_'
        'carbon',
        units='kg m-2 s-1',
    )
    return cube


def test_get_gpp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CanESM5', 'Lmon', 'gpp')
    assert fix == [Gpp(None)]


def test_gpp_fix_data(gpp_cube):
    """Test ``fix_data`` for ``gpp``."""
    fix = Gpp(None)
    out_cube = fix.fix_data(gpp_cube)
    np.testing.assert_allclose(out_cube.data,
                               np.ma.masked_invalid([np.nan, 1]))
    assert np.array_equal(out_cube.data.mask, [True, False])
