"""Tests for the fixes of MIROC-MIROC5."""
import pytest
import iris

from esmvalcore.cmor._fixes.cordex.miroc_miroc5 import wrf361h
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord([0.0, 1.0],
                                              var_name='time',
                                              standard_name='time',
                                              long_name='time')
    wrong_height_coord = iris.coords.DimCoord([2.0],
                                              var_name='height')
    wrong_cube = iris.cube.Cube(
        [[10.0], [10.0]],
        var_name='tas',
        dim_coords_and_dims=[
            (correct_time_coord, 0),
            (wrong_height_coord, 1)],
    )
    return iris.cube.CubeList([wrong_cube])


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_cclm4_8_17fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'CCLM4-8-17',
        'Amon',
        short_name,
        extra_facets={'driver': 'MIROC-MIROC5'})
    assert isinstance(fix[0], Fix)


def test_get_remo2015_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'REMO2015',
        'Amon',
        'pr',
        extra_facets={'driver': 'MIROC-MIROC5'})
    assert isinstance(fix[0], Fix)


def test_get_wrf361h_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'WRF361H',
        'Amon',
        'tas',
        extra_facets={'driver': 'MIROC-MIROC5'})
    assert isinstance(fix[0], Fix)


def test_wrf361h_height_fix(cubes):
    fix = wrf361h.Tas(None)
    out_cubes = fix.fix_metadata(cubes)
    for cube in out_cubes:
        assert cube.ndim == 1
