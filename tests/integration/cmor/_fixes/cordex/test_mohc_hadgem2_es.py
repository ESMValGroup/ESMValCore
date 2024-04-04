"""Tests for the fixes for driver MOHC-HadGEM2-ES."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.mohc_hadgem2_es import hirham5, wrf381p
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord([0.0],
                                              var_name='time',
                                              standard_name='time',
                                              long_name='time')
    wrong_time_coord = iris.coords.DimCoord([0.0],
                                            var_name='time',
                                            standard_name='time',
                                            long_name='wrong')
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name='latitude',
                                             long_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='latitude',
                                           long_name='latitude',
                                           attributes={'wrong': 'attr'})
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude',
                                             long_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude',
                                           long_name='longitude',
                                           attributes={'wrong': 'attr'})
    correct_cube = iris.cube.Cube(
        [[[10.0]]],
        var_name='tas',
        dim_coords_and_dims=[
            (correct_time_coord, 0),
            (correct_lat_coord, 1),
            (correct_lon_coord, 2)],
    )
    wrong_cube = iris.cube.Cube(
        [[[10.0]]],
        var_name='tas',
        dim_coords_and_dims=[
            (wrong_time_coord, 0),
            (wrong_lat_coord, 1),
            (wrong_lon_coord, 2)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_hirham5_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'HIRHAM5',
        'Amon',
        'pr',
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_remo2015_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'REMO2015',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_rca4_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'RCA4',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


def test_hirham5_fix(cubes):
    fix = hirham5.Pr(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('latitude').attributes == {}
        assert cube.coord('longitude').attributes == {}


@pytest.mark.parametrize(
    'short_name',
    ['tasmax', 'tasmin', 'tas', 'hurs', 'huss'])
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'WRF381P',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
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
