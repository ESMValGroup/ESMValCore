"""Tests for the fixes of AWI-ESM-1-1-LR."""
import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.awi_esm_1_1_lr import AllVars
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def sample_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'AWI-ESM-1-1-LR', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_allvars_fix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 0001-01-01 00:00:00'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes[
            'parent_time_units'] == 'days since 0001-01-01 00:00:00'


def test_allvars_no_need_tofix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 0001-01-01 00:00:00'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes[
            'parent_time_units'] == 'days since 0001-01-01 00:00:00'
