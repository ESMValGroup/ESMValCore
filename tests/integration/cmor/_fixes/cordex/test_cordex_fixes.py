"""Tests for general CORDEX fixes."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    CLMcomCCLM4817,
    MOHCHadREM3GA705,
    TimeLongName,
)


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
                                           long_name='latitude')
    correct_lon_coord = iris.coords.DimCoord([0.0],
                                             var_name='lon',
                                             standard_name='longitude',
                                             long_name='longitude')
    wrong_lon_coord = iris.coords.DimCoord([0.0],
                                           var_name='longitudeCoord',
                                           standard_name='longitude',
                                           long_name='longitude')
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


def test_mohchadrem3ga705_fix_metadata(cubes):
    fix = MOHCHadREM3GA705(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        for coord in cube.coords():
            right_coord = cubes[0].coord(coord)
            assert coord == right_coord


def test_timelongname_fix_metadata(cubes):
    fix = TimeLongName(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('time').long_name == 'time'


def test_clmcomcclm4817_fix_metadata(cubes):
    cubes[0].coord('time').units = Unit(
        'days since 1850-1-1 00:00:00',
        calendar='proleptic_gregorian')
    cubes[1].coord('time').units = Unit(
        'days since 1850-1-1 00:00:00',
        calendar='gregorian')
    for coord in cubes[1].coords():
        coord.points = coord.core_points().astype(
            '>f8', casting='same_kind')

    fix = CLMcomCCLM4817(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('time').units == Unit(
            'days since 1850-1-1 00:00:00',
            calendar='proleptic_gregorian')
        for coord in cube.coords():
            assert coord.points.dtype == np.float64
