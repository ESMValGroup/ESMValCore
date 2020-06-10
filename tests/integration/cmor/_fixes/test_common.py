"""Test for common fixes used for multiple datasets."""
import os

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.common import (ClFixHybridHeightCoord,
                                           ClFixHybridPressureCoord)
from esmvalcore.cmor.table import get_var_info
from esmvalcore.iris_helpers import var_name_constraint


def create_hybrid_pressure_file_without_ap(dataset, short_name):
    """Create dataset without vertical auxiliary coordinate ``ap``."""
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=2)
    dataset.createDimension('lat', size=3)
    dataset.createDimension('lon', size=4)
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
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].units = '1'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lat'][:] = [-30.0, 0.0, 30.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0, 90.0, 120.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0]]
    dataset.variables['ps'][:] = np.arange(1 * 3 * 4).reshape(1, 3, 4)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'
    dataset.variables['ps'].additional_attribute = 'xyz'

    # Variable
    dataset.createVariable(short_name, np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables[short_name][:] = np.full((1, 2, 3, 4), 0.0,
                                               dtype=np.float32)
    dataset.variables[short_name].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables[short_name].units = '%'


def create_hybrid_pressure_file_with_a(dataset, short_name):
    """Create netcdf file with issues in hybrid pressure coordinate."""
    create_hybrid_pressure_file_without_ap(dataset, short_name)
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('p0', np.float64, dimensions=())
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['p0'][:] = 1.0
    dataset.variables['p0'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'p0: p0 a: a b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')


def create_hybrid_pressure_file_with_ap(dataset, short_name):
    """Create netcdf file with issues in hybrid pressure coordinate."""
    create_hybrid_pressure_file_without_ap(dataset, short_name)
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['ap'][:] = [1.0, 2.0]
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['ap'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'ap: ap b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'ap: ap_bnds b: b_bnds ps: ps')


@pytest.fixture
def cl_file_with_a(tmp_path):
    """Create netcdf file for ``cl`` with ``a`` coordinate."""
    nc_path = os.path.join(tmp_path, 'cl_a.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_a(dataset, 'cl')
    dataset.close()
    return nc_path


@pytest.fixture
def cl_file_with_ap(tmp_path):
    """Create netcdf file for ``cl`` with ``ap`` coordinate."""
    nc_path = os.path.join(tmp_path, 'cl_ap.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_ap(dataset, 'cl')
    dataset.close()
    return nc_path


@pytest.fixture
def cli_file_with_a(tmp_path):
    """Create netcdf file for ``cli`` with ``a`` coordinate."""
    nc_path = os.path.join(tmp_path, 'cli_a.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_a(dataset, 'cli')
    dataset.close()
    return nc_path


@pytest.fixture
def cli_file_with_ap(tmp_path):
    """Create netcdf file for ``cli`` with ``ap`` coordinate."""
    nc_path = os.path.join(tmp_path, 'cli_ap.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_ap(dataset, 'cli')
    dataset.close()
    return nc_path


@pytest.fixture
def clw_file_with_a(tmp_path):
    """Create netcdf file for ``clw`` with ``a`` coordinate."""
    nc_path = os.path.join(tmp_path, 'clw_a.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_a(dataset, 'clw')
    dataset.close()
    return nc_path


@pytest.fixture
def clw_file_with_ap(tmp_path):
    """Create netcdf file for ``clw`` with ``ap`` coordinate."""
    nc_path = os.path.join(tmp_path, 'clw_ap.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_pressure_file_with_ap(dataset, 'clw')
    dataset.close()
    return nc_path


AIR_PRESSURE_POINTS = np.array([[[[1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0]],
                                 [[2.0, 3.0, 4.0, 5.0],
                                  [6.0, 7.0, 8.0, 9.0],
                                  [10.0, 11.0, 12.0, 13.0]]]])
AIR_PRESSURE_BOUNDS = np.array([[[[[0.0, 1.5],
                                   [-1.0, 2.0],
                                   [-2.0, 2.5],
                                   [-3.0, 3.0]],
                                  [[-4.0, 3.5],
                                   [-5.0, 4.0],
                                   [-6.0, 4.5],
                                   [-7.0, 5.0]],
                                  [[-8.0, 5.5],
                                   [-9.0, 6.0],
                                   [-10.0, 6.5],
                                   [-11.0, 7.0]]],
                                 [[[1.5, 3.0],
                                   [2.0, 5.0],
                                   [2.5, 7.0],
                                   [3.0, 9.0]],
                                  [[3.5, 11.0],
                                   [4.0, 13.0],
                                   [4.5, 15.0],
                                   [5.0, 17.0]],
                                  [[5.5, 19.0],
                                   [6.0, 21.0],
                                   [6.5, 23.0],
                                   [7.0, 25.0]]]]])


def hybrid_pressure_coord_fix_metadata(nc_path, short_name, fix):
    """Test ``fix_metadata`` of file with hybrid pressure coord."""
    cubes = iris.load(nc_path)

    # Raw cubes
    assert len(cubes) == 4
    var_names = [cube.var_name for cube in cubes]
    assert short_name in var_names
    assert 'ps' in var_names
    assert 'b_bnds' in var_names

    # Raw cube
    cube = cubes.extract_strict(var_name_constraint(short_name))
    air_pressure_coord = cube.coord('air_pressure')
    assert air_pressure_coord.points is not None
    assert air_pressure_coord.bounds is None
    np.testing.assert_allclose(air_pressure_coord.points, AIR_PRESSURE_POINTS)

    # Raw ps cube
    ps_cube = cubes.extract_strict('surface_air_pressure')
    assert ps_cube.attributes == {'additional_attribute': 'xyz'}

    # Apply fix
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_strict(var_name_constraint(short_name))
    fixed_air_pressure_coord = fixed_cube.coord('air_pressure')
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(fixed_air_pressure_coord.points,
                               AIR_PRESSURE_POINTS)
    np.testing.assert_allclose(fixed_air_pressure_coord.bounds,
                               AIR_PRESSURE_BOUNDS)
    surface_pressure_coord = fixed_cube.coord(var_name='ps')
    assert surface_pressure_coord.attributes == {}

    return var_names


def test_cl_hybrid_pressure_coord_fix_metadata_with_a(cl_file_with_a):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    var_names = hybrid_pressure_coord_fix_metadata(
        cl_file_with_a, 'cl', ClFixHybridPressureCoord(vardef))
    assert 'a_bnds' in var_names


def test_cl_hybrid_pressure_coord_fix_metadata_with_ap(cl_file_with_ap):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    var_names = hybrid_pressure_coord_fix_metadata(
        cl_file_with_ap, 'cl', ClFixHybridPressureCoord(vardef))
    assert 'ap_bnds' in var_names


def create_hybrid_height_file(dataset, short_name):
    """Create dataset with hybrid height coordinate."""
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

    # Variable
    dataset.createVariable(short_name, np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables[short_name][:] = np.full((1, 2, 1, 2), 0.0,
                                               dtype=np.float32)
    dataset.variables[short_name].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables[short_name].units = '%'


@pytest.fixture
def cl_file_with_height(tmp_path):
    """Create netcdf file for ``cl`` with hybrid height coordinate."""
    nc_path = os.path.join(tmp_path, 'cl_hybrid_height.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_height_file(dataset, 'cl')
    dataset.close()
    return nc_path


@pytest.fixture
def cli_file_with_height(tmp_path):
    """Create netcdf file for ``cli`` with hybrid height coordinate."""
    nc_path = os.path.join(tmp_path, 'cli_hybrid_height.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_height_file(dataset, 'cli')
    dataset.close()
    return nc_path


@pytest.fixture
def clw_file_with_height(tmp_path):
    """Create netcdf file for ``clw`` with hybrid height coordinate."""
    nc_path = os.path.join(tmp_path, 'clw_hybrid_height.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_height_file(dataset, 'clw')
    dataset.close()
    return nc_path


HEIGHT_POINTS = np.array([[[1.0, 1.0]],
                          [[2.0, 3.0]]])
HEIGHT_BOUNDS_WRONG = np.array([[[[0.5, 1.5],
                                  [0.5, 1.5]]],
                                [[[1.5, 3.0],
                                  [2.5, 4.0]]]])
HEIGHT_BOUNDS_RIGHT = np.array([[[[0.5, 1.5],
                                  [-0.5, 2.0]]],
                                [[[1.5, 3.0],
                                  [2.0, 5.0]]]])
PRESSURE_POINTS = np.array([[[101312.98512207, 101312.98512207]],
                            [[101300.97123885, 101288.95835383]]])
PRESSURE_BOUNDS = np.array([[[[101318.99243691, 101306.9780559],
                              [101331.00781103, 101300.97123885]]],
                            [[[101306.9780559, 101288.95835383],
                              [101300.97123885, 101264.93559234]]]])


def hybrid_height_coord_fix_metadata(nc_path, short_name, fix):
    """Test ``fix_metadata`` of file with hybrid height coord."""
    cubes = iris.load(nc_path)

    # Raw cubes
    assert len(cubes) == 3
    var_names = [cube.var_name for cube in cubes]
    assert short_name in var_names
    assert 'orog' in var_names
    assert 'b_bnds' in var_names

    # Raw cube
    cube = cubes.extract_strict(var_name_constraint(short_name))
    height_coord = cube.coord('altitude')
    assert height_coord.points is not None
    assert height_coord.bounds is not None
    np.testing.assert_allclose(height_coord.points, HEIGHT_POINTS)
    np.testing.assert_allclose(height_coord.bounds, HEIGHT_BOUNDS_WRONG)
    assert not np.allclose(height_coord.bounds, HEIGHT_BOUNDS_RIGHT)
    assert not cube.coords('air_pressure')

    # Apply fix
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_strict(var_name_constraint(short_name))
    fixed_height_coord = fixed_cube.coord('altitude')
    assert fixed_height_coord.points is not None
    assert fixed_height_coord.bounds is not None
    np.testing.assert_allclose(fixed_height_coord.points, HEIGHT_POINTS)
    np.testing.assert_allclose(fixed_height_coord.bounds, HEIGHT_BOUNDS_RIGHT)
    assert not np.allclose(fixed_height_coord.bounds, HEIGHT_BOUNDS_WRONG)
    air_pressure_coord = cube.coord('air_pressure')
    np.testing.assert_allclose(air_pressure_coord.points, PRESSURE_POINTS)
    np.testing.assert_allclose(air_pressure_coord.bounds, PRESSURE_BOUNDS)
    assert air_pressure_coord.var_name == 'plev'
    assert air_pressure_coord.standard_name == 'air_pressure'
    assert air_pressure_coord.long_name == 'pressure'
    assert air_pressure_coord.units == 'Pa'


def test_cl_hybrid_height_coord_fix_metadata(cl_file_with_height):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    hybrid_height_coord_fix_metadata(cl_file_with_height, 'cl',
                                     ClFixHybridHeightCoord(vardef))
