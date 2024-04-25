"""Tests for the fixes for driver CNRM-CERFACS-CNRM-CM5."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5 import (
    aladin63,
    wrf381p,)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord([0.0],
                                              var_name='time',
                                              standard_name='time',
                                              long_name='time')
    correct_height_coord = iris.coords.AuxCoord([2.0],
                                                var_name='height')
    wrong_height_coord = iris.coords.AuxCoord([10.0],
                                              var_name='height')
    correct_cube = iris.cube.Cube(
        [10.0],
        var_name='tas',
        dim_coords_and_dims=[(correct_time_coord, 0)],
        aux_coords_and_dims=[(correct_height_coord, ())]
    )
    wrong_cube = iris.cube.Cube(
        [10.0],
        var_name='tas',
        dim_coords_and_dims=[(correct_time_coord, 0)],
        aux_coords_and_dims=[(wrong_height_coord, ())]
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_aladin63_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'ALADIN63',
        'Amon',
        short_name,
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'})
    assert isinstance(fix[0], Fix)


def test_aladin63_height_fix(cubes):
    fix = aladin63.Tas(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('height').points == 2.0


@pytest.mark.parametrize(
    'short_name',
    ['tasmax', 'tasmin', 'tas', 'hurs', 'huss'])
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'WRF381P',
        'Amon',
        short_name,
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'})
    assert isinstance(fix[0], Fix)


def test_wrf381p_height_fix():
    time_coord = iris.coords.DimCoord([0.0],
                                      var_name='time',
                                      standard_name='time',
                                      long_name='time')
    cube = iris.cube.Cube(
        [10.0],
        var_name='tas',
        dim_coords_and_dims=[(time_coord, 0)],
    )
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = wrf381p.Tas(vardef)
    out_cubes = fix.fix_metadata([cube])
    assert out_cubes[0].coord('height').points == 2.0
