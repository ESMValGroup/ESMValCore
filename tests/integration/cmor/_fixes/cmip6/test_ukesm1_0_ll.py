"""Tests for the fixes of UKESM1-0-LL."""
import os

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.ukesm1_0_ll import AllVars, Cl
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def sample_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'UKESM1-0-LL', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_allvars_fix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'


def test_allvars_no_need_tofix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'UKESM1-0-LL', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


@pytest.fixture
def cl_file(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'ukesm1_0_ll_cl.nc')
    dataset = Dataset(nc_path, mode='w')

    # Dimensions
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=2)
    dataset.createDimension('lat', size=1)
    dataset.createDimension('lon', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = [0.0]
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 6543-2-1'
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_height_coordinate')
    dataset.variables['lev'].units = 'm'
    dataset.variables['lev'].formula_terms = 'a: lev b: b orog: orog'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_height_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_terms = (
        'a: lev_bnds b: b_bnds orog: orog')
    dataset.variables['lat'][:] = [0.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of height coordinate
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('orog', np.float64, dimensions=('lat', 'lon'))
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0]]
    dataset.variables['orog'][:] = [[0.0, 1.0]]
    dataset.variables['orog'].standard_name = 'surface_altitude'
    dataset.variables['orog'].units = 'm'

    # Cl variable
    dataset.createVariable('cl', np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables['cl'][:] = np.full((1, 2, 1, 2), 0.0, dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'

    dataset.close()
    return nc_path


def test_cl_fix_metadata(cl_file):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = iris.load(cl_file)

    # Raw cubes
    assert len(cubes) == 3
    var_names = [cube.var_name for cube in cubes]
    assert 'cl' in var_names
    assert 'orog' in var_names
    assert 'b_bnds' in var_names

    # Height coordinate
    height_points = np.array([[[1.0, 1.0]],
                              [[2.0, 3.0]]])
    height_bounds_wrong = np.array([[[[0.5, 1.5],
                                      [0.5, 1.5]]],
                                    [[[1.5, 3.0],
                                      [2.5, 4.0]]]])
    height_bounds_right = np.array([[[[0.5, 1.5],
                                      [-0.5, 2.0]]],
                                    [[[1.5, 3.0],
                                      [2.0, 5.0]]]])
    air_pressure_points = np.array([[[101312.98512207, 101312.98512207]],
                                    [[101300.97123885, 101288.95835383]]])
    air_pressure_bounds = np.array([[[[101318.99243691, 101306.9780559],
                                      [101331.00781103, 101300.97123885]]],
                                    [[[101306.9780559, 101288.95835383],
                                      [101300.97123885, 101264.93559234]]]])

    # Raw cl cube
    cl_cube = cubes.extract_strict('cloud_area_fraction_in_atmosphere_layer')
    height_coord = cl_cube.coord('altitude')
    assert height_coord.points is not None
    assert height_coord.bounds is not None
    np.testing.assert_allclose(height_coord.points, height_points)
    np.testing.assert_allclose(height_coord.bounds, height_bounds_wrong)
    assert not np.allclose(height_coord.bounds, height_bounds_right)
    assert not cl_cube.coords('air_pressure')

    # Apply fix
    fix = Cl(None)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    fixed_height_coord = fixed_cl_cube.coord('altitude')
    assert fixed_height_coord.points is not None
    assert fixed_height_coord.bounds is not None
    np.testing.assert_allclose(fixed_height_coord.points, height_points)
    np.testing.assert_allclose(fixed_height_coord.bounds, height_bounds_right)
    assert not np.allclose(fixed_height_coord.bounds, height_bounds_wrong)
    air_pressure_coord = cl_cube.coord('air_pressure')
    np.testing.assert_allclose(air_pressure_coord.points, air_pressure_points)
    np.testing.assert_allclose(air_pressure_coord.bounds, air_pressure_bounds)
    assert air_pressure_coord.var_name == 'plev'
    assert air_pressure_coord.standard_name == 'air_pressure'
    assert air_pressure_coord.long_name == 'pressure'
    assert air_pressure_coord.units == 'Pa'
