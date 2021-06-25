"""Tests for the fixes of MCM-UA-1-0."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.mcm_ua_1_0 import AllVars, Omon, Tas, Uas
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name=' latitude  ',
                                             long_name='  latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='  latitude',
                                           long_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='  longitude  ',
                                             long_name='longitude  ')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude',
                                           long_name='  longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='tas',
        standard_name='air_temperature   ',
        long_name='   Air Temperature   ',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        standard_name='   air_temperature   ',
        long_name='Air Temperature',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={'parent_time_units': 'days since 0000-00-00 (noleap)'},
    )
    scalar_cube = iris.cube.Cube(0.0, var_name='ps',
                                 standard_name='air_pressure   ',
                                 long_name=' Air pressure  ')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


@pytest.fixture
def uas_cubes():
    correct_lat_coord = iris.coords.DimCoord([0.0],
                                             var_name='lat',
                                             standard_name=' latitude  ',
                                             long_name='  latitude')
    wrong_lat_coord = iris.coords.DimCoord([0.0],
                                           var_name='latitudeCoord',
                                           standard_name='  latitude',
                                           long_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='  longitude  ',
                                             long_name='longitude  ')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude',
                                           long_name='  longitude')
    correct_cube = iris.cube.Cube(
        [[10.0]],
        var_name='uas',
        standard_name='eastward_wind   ',
        long_name='   East Near-Surface Wind   ',
        dim_coords_and_dims=[(correct_lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0]],
        var_name='ta',
        standard_name='   air_temperature   ',
        long_name='Air Temperature',
        dim_coords_and_dims=[(wrong_lat_coord, 0), (wrong_lon_coord, 1)],
        attributes={'parent_time_units': 'days since 0000-00-00 (noleap)'},
    )
    scalar_cube = iris.cube.Cube(0.0, var_name='ps',
                                 standard_name='air_pressure   ',
                                 long_name=' Air pressure  ')
    return iris.cube.CubeList([correct_cube, wrong_cube, scalar_cube])


@pytest.fixture
def cubes_bounds():
    lat_coord = iris.coords.DimCoord([0.0],
                                     var_name='lat',
                                     standard_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0, 356.25],
                                             bounds=[[-1.875, 1.875],
                                                     [354.375, 358.125]],
                                             var_name='lon',
                                             standard_name='longitude',
                                             circular=True)
    wrong_lon_coord = iris.coords.DimCoord([0, 356.25],
                                           bounds=[[-1.875, 1.875],
                                                   [354.375, 360]],
                                           var_name='lon',
                                           standard_name='longitude')
    correct_cube = iris.cube.Cube(
        [[10.0, 10.0]],
        var_name='tas',
        dim_coords_and_dims=[(lat_coord, 0), (correct_lon_coord, 1)],
    )
    wrong_cube = iris.cube.Cube(
        [[10.0, 10.0]],
        var_name='tas',
        dim_coords_and_dims=[(lat_coord, 0), (wrong_lon_coord, 1)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_allvars_fix():
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'Amon',
                        'arbitrary_var_name_and_wrong_lon_bnds')
    assert fix == [AllVars(None)]


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'Amon', 'tas')
    assert fix == [Tas(None), AllVars(None)]


def test_get_uas_fix():
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'Amon', 'uas')
    assert fix == [Uas(None), AllVars(None)]


def test_allvars_fix_metadata(cubes):
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        if cube.var_name == 'ps':
            assert cube.standard_name == 'air_pressure'
            assert cube.long_name == 'Air pressure'
        elif cube.var_name == 'tas' or cube.var_name == 'ta':
            assert cube.standard_name == 'air_temperature'
            assert cube.long_name == 'Air Temperature'
        else:
            assert False, "Invalid var_name"
        try:
            lat_coord = cube.coord('latitude')
        except iris.exceptions.CoordinateNotFoundError:
            assert cube.var_name == 'ps'
        else:
            assert lat_coord.var_name == 'lat'
            assert lat_coord.standard_name == 'latitude'
            assert lat_coord.long_name == 'latitude'
        try:
            lon_coord = cube.coord('longitude')
        except iris.exceptions.CoordinateNotFoundError:
            assert cube.var_name == 'ps'
        else:
            assert lon_coord.var_name == 'lon'
            assert lon_coord.standard_name == 'longitude'
            assert lon_coord.long_name == 'longitude'
        if 'parent_time_units' in cube.attributes:
            assert cube.attributes['parent_time_units'] == (
                'days since 0000-00-00')


def test_allvars_fix_lon_bounds(cubes_bounds):
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes_bounds)
    assert cubes_bounds is out_cubes
    for cube in out_cubes:
        try:
            lon_coord = cube.coord('longitude')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            assert lon_coord.bounds[-1][-1] == 358.125
            assert lon_coord.circular


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
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Tas(vardef)

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


def test_uas_fix_metadata(uas_cubes):
    for cube in uas_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord('height')
    height_coord = iris.coords.AuxCoord(10.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    vardef = get_var_info('CMIP6', 'Amon', 'uas')
    fix = Uas(vardef)

    # Check fix
    out_cubes = fix.fix_metadata(uas_cubes)
    assert out_cubes[0].var_name == 'uas'
    coord = out_cubes[0].coord('height')
    assert coord == height_coord

    # Check that height coordinate is not added twice
    out_cubes_2 = fix.fix_metadata(out_cubes)
    assert out_cubes_2[0].var_name == 'uas'
    coord = out_cubes_2[0].coord('height')
    assert coord == height_coord


@pytest.fixture
def thetao_cubes():
    time_coord = iris.coords.DimCoord(
        [0.0004, 1.09776], var_name='time', standard_name='time',
        units='days since 1850-01-01 00:00:00')
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lat', standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord([-0.9375, 357.1875],
                                     bounds=[[-1.875, 0.], [356.25, 358.125]],
                                     var_name='lon',
                                     standard_name='longitude')
    lev_coord = iris.coords.DimCoord(
        [5.0, 10.0], bounds=[[2.5, 7.5], [7.5, 12.5]],
        var_name='lev', standard_name=None, units='m',
        attributes={'positive': 'up'})
    coord_specs = [
        (time_coord, 0),
        (lev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    thetao_cube = iris.cube.Cube(
        np.arange(16).reshape(2, 2, 2, 2),
        var_name='thetao',
        dim_coords_and_dims=coord_specs,
    )

    return iris.cube.CubeList([thetao_cube])


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MCM-UA-1-0', 'Omon', 'thetao')
    assert fix == [Omon(None), AllVars(None)]


def test_thetao_fix_metadata(thetao_cubes):
    """Test ``fix_metadata`` for ``thetao``."""
    vardef = get_var_info('CMIP6', 'Omon', 'thetao')
    fix_omon = Omon(vardef)
    fix_allvars = AllVars(vardef)
    out_cubes = fix_omon.fix_metadata(thetao_cubes)
    out_cubes = fix_allvars.fix_metadata(out_cubes)
    assert out_cubes is thetao_cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check data of cube
    np.testing.assert_allclose(out_cube.data,
                               [[[[1, 0],
                                  [3, 2]],
                                 [[5, 4],
                                  [7, 6]]],
                                [[[9, 8],
                                  [11, 10]],
                                 [[13, 12],
                                  [15, 14]]]])

    # Check data of longitude
    lon_coord = out_cube.coord('longitude')
    np.testing.assert_allclose(lon_coord.points, [357.1875, 359.0625])
    np.testing.assert_allclose(lon_coord.bounds,
                               [[356.25, 358.125], [358.125, 360.0]])

    # Check metadata of depth coordinate
    depth_coord = out_cube.coord('depth')
    assert depth_coord.standard_name == 'depth'
    assert depth_coord.var_name == 'lev'
    assert depth_coord.long_name == 'ocean depth coordinate'
    assert depth_coord.units == 'm'
    assert depth_coord.attributes == {'positive': 'down'}
