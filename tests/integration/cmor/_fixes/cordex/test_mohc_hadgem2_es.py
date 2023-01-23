"""Tests for the fixes for driver MOHC-HadGEM2-ES."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.mohc_hadgem2_es import dmi_hirham5
from esmvalcore.cmor.fix import Fix


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


def test_get_dmi_hirham5_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'DMI-HIRHAM5',
        'Amon',
        'pr',
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_gerics_remo2015_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'GERICS-REMO2015',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_mohc_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'MOHC-HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_smhi_rca4_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'SMHI-RCA4',
        'Amon',
        short_name,
        extra_facets={'driver': 'MOHC-HadGEM2-ES'})
    assert isinstance(fix[0], Fix)


def test_dmi_hirham5_fix(cubes):
    fix = dmi_hirham5.Pr(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('latitude').attributes == {}
        assert cube.coord('longitude').attributes == {}
