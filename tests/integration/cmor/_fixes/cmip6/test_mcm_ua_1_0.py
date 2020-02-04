"""Tests for the fixes of MCM-UA-1-0."""
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.mcm_ua_1_0 import AllVars, Tas
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0], var_name='lat',
                                             standard_name='latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0], var_name='latitudeCoord',
                                           standard_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0], var_name='lon',
                                             standard_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0], var_name='longitudeCoord',
                                           standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='tas',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={'parent_time_units': 'days since 0000-00-00 (noleap)'},
    )
    scalar_cube = iris.cube.Cube(0.0, var_name='ps')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


def test_get_allvars_fix():
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'arbitrary_var_name')
    assert fix == [AllVars()]


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'tas')
    assert fix == [AllVars(), Tas()]


def test_allvars_fix_metadata(cubes):
    fix = AllVars()
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        try:
            lat_coord = cube.coord('latitude')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            assert lat_coord.var_name == 'lat'
            assert lat_coord.standard_name == 'latitude'
        try:
            lon_coord = cube.coord('longitude')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            assert lon_coord.var_name == 'lon'
            assert lon_coord.standard_name == 'longitude'
        if 'parent_time_units' in cube.attributes:
            assert cube.attributes['parent_time_units'] == (
                'days since 0000-00-00')


def test_tas_fix_metadata(cubes):
    for cube in cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord('height')
    height_coord = iris.coords.AuxCoord(2.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    fix = Tas()

    # Check fix
    out_cubes = fix.fix_metadata(cubes)
    assert out_cubes[0].var_name == 'tas'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == 'tas'
    coord = out_cubes_2[0].coord('height')
    assert coord == height_coord
