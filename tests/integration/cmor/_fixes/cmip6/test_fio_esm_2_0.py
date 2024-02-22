"""Tests for the fixes of FIO-ESM-2-0."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.fio_esm_2_0 import Amon, Omon, Tos
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_clt_fix():
    """Test `Clt.fix_data`."""
    cube = iris.cube.Cube(0.5)
    fix = Fix.get_fixes('CMIP6', 'FIO-ESM-2-0', 'Amon', 'clt')[0]
    out_cube = fix.fix_data(cube)
    np.testing.assert_allclose(out_cube.data, 50.0)
    assert out_cube.units == '%'


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FIO-ESM-2-0', 'Amon', 'tas')
    assert fix == [Amon(None), GenericFix(None)]


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FIO-ESM-2-0', 'Omon', 'tos')
    assert fix == [OceanFixGrid(None), Omon(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid


@pytest.fixture
def tas_cubes():
    correct_time_coord = iris.coords.DimCoord(
        [15.5, 45, 74.5],
        bounds=[[0., 31.], [31., 59.], [59., 90.]],
        var_name='time',
        standard_name='time',
        units=Unit('days since 0001-01-01 00:00:00', calendar='365_day'))

    correct_lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                             bounds=[[-0.5, 0.5], [0.5, 1.5]],
                                             var_name='lat',
                                             standard_name='latitude',
                                             units='degrees')

    wrong_lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                           bounds=[[-0.5, 0.5], [1.5, 2.]],
                                           var_name='lat',
                                           standard_name='latitude',
                                           units='degrees')

    correct_lon_coord = iris.coords.DimCoord([0.0, 1.0],
                                             bounds=[[-0.5, 0.5], [0.5, 1.5]],
                                             var_name='lon',
                                             standard_name='longitude',
                                             units='degrees')

    wrong_lon_coord = iris.coords.DimCoord([0.0, 1.0],
                                           bounds=[[-0.5, 0.5], [1.5, 2.]],
                                           var_name='lon',
                                           standard_name='longitude',
                                           units='degrees')

    correct_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                  var_name='tas',
                                  dim_coords_and_dims=[(correct_time_coord, 0),
                                                       (correct_lat_coord, 1),
                                                       (correct_lon_coord, 2)],
                                  attributes={'table_id': 'Amon'},
                                  units=Unit('degC'))

    wrong_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                var_name='tas',
                                dim_coords_and_dims=[(correct_time_coord, 0),
                                                     (wrong_lat_coord, 1),
                                                     (wrong_lon_coord, 2)],
                                attributes={'table_id': 'Amon'},
                                units=Unit('degC'))

    return iris.cube.CubeList([correct_cube, wrong_cube])


@pytest.fixture
def tos_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.2],
                                      standard_name='time',
                                      var_name='time',
                                      units='days since 1850-01-01')
    lat_coord = iris.coords.DimCoord([23.0211555789],
                                     standard_name='latitude',
                                     var_name='lat',
                                     units='degrees_north')
    lon_coord = iris.coords.DimCoord([30.0211534556],
                                     standard_name='longitude',
                                     var_name='lon',
                                     units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0]]],
                          standard_name='sea_surface_temperature',
                          var_name='tos',
                          units='degC',
                          dim_coords_and_dims=coords_specs)
    return iris.cube.CubeList([cube])


def test_tos_fix_metadata(tos_cubes, caplog):
    """Test ``fix_metadata``."""
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = Omon(vardef, extra_facets={'dataset': 'FIO-ESM-2-0'})
    fixed_cubes = fix.fix_metadata(tos_cubes)
    assert len(fixed_cubes) == 1
    fixed_tos_cube = fixed_cubes.extract_cube('sea_surface_temperature')
    fixed_lon = fixed_tos_cube.coord('longitude')
    fixed_lat = fixed_tos_cube.coord('latitude')
    np.testing.assert_equal(fixed_lon.points, [30.021153])
    np.testing.assert_equal(fixed_lat.points, [23.021156])
    msg = ("Using 'area_weighted' regridder scheme in Omon variables "
           "for dataset FIO-ESM-2-0 causes discontinuities in the longitude "
           "coordinate.")
    assert msg in caplog.text


def test_amon_fix_metadata(tas_cubes):
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Amon(vardef)
    out_cubes = fix.fix_metadata(tas_cubes)
    assert tas_cubes is out_cubes
    for cube in out_cubes:
        time = cube.coord('time')
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
        assert all(time.bounds[1:, 0] == time.bounds[:-1, 1])
        assert all(lat.bounds[1:, 0] == lat.bounds[:-1, 1])
        assert all(lon.bounds[1:, 0] == lon.bounds[:-1, 1])
