"""Tests for the fixes for driver CNRM-CERFACS-CNRM-CM5."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5 import cnrm_aladin63
from esmvalcore.cmor.fix import Fix


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
def test_get_mohc_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'MOHC-HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_cnrm_aladin63_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'CNRM-ALADIN63',
        'Amon',
        short_name,
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'})
    assert isinstance(fix[0], Fix)


def test_cnrm_aladin63_height_fix(cubes):
    fix = cnrm_aladin63.Tas(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('height').points == 2.0
