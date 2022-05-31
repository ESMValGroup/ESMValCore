"""Tests for the fixes of FGOALS-f3-L."""

import numpy as np
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.fgoals_f3_l import AllVars, Tos
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord(
        [15.5, 45, 74.5],
        bounds=[[0., 31.], [31., 59.], [59., 90.]],
        var_name='time',
        standard_name='time',
        units=Unit('days since 0001-01-01 00:00:00', calendar='365_day'))
    wrong_time_coord = iris.coords.DimCoord(
        [15.5, 45, 74.5],
        bounds=[[5.5, 25.5], [35., 55.], [64.5, 84.5]],
        var_name='time',
        standard_name='time',
        units=Unit('days since 0001-01-01 00:00:00', calendar='365_day'))

    correct_lat_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        bounds=[[-0.5, 0.5], [0.5, 1.5]],
        var_name='lat',
        standard_name='latitude',
        units='degrees')

    wrong_lat_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        bounds=[[-0.5, 0.5], [1.5, 2.]],
        var_name='lat',
        standard_name='latitude',
        units='degrees')

    correct_lon_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        bounds=[[-0.5, 0.5], [0.5, 1.5]],
        var_name='lon',
        standard_name='longitude',
        units='degrees')

    wrong_lon_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        bounds=[[-0.5, 0.5], [1.5, 2.]],
        var_name='lon',
        standard_name='longitude',
        units='degrees')

    correct_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                  var_name='tas',
                                  dim_coords_and_dims=[(correct_time_coord, 0),
                                                       (correct_lat_coord, 1),
                                                       (correct_lon_coord, 2)
                                                       ],
                                  attributes={'table_id': 'Amon'},
                                  units=Unit('degC'))

    wrong_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                var_name='tas',
                                dim_coords_and_dims=[(wrong_time_coord, 0),
                                                     (wrong_lat_coord, 1),
                                                     (wrong_lon_coord, 2)],
                                attributes={'table_id': 'Amon'},
                                units=Unit('degC'))

    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_allvars_fix():
    fix = Fix.get_fixes('CMIP6', 'FGOALS-f3-L', 'Amon', 'wrong_time_bnds')
    assert fix == [AllVars(None)]


def test_allvars_fix_metadata(cubes):
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        time = cube.coord('time')
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
        assert all(time.bounds[1:, 0] == time.bounds[:-1, 1])
        assert all(lat.bounds[1:, 0] == lat.bounds[:-1, 1])
        assert all(lon.bounds[1:, 0] == lon.bounds[:-1, 1])


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid
