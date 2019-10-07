"""Tests for the fixes of UKESM1-0-LL."""
import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.ukesm1_0_ll import AllVars
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def sample_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'UKESM1-0-LL', 'tas')
    assert fix == [AllVars()]


def test_allvars_fix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars().fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'


def test_allvars_no_need_tofix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars().fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'
