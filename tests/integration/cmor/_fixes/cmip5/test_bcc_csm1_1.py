"""Test bcc-csm1-1 fixes."""
import os
import unittest

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1 import Cl, Tos
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'bcc-csm1-1', 'Amon', 'cl')
    assert fix == [Cl(None)]


def create_cl_file_without_ap(dataset):
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

    # Cl variable
    dataset.createVariable('cl', np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables['cl'][:] = np.full((1, 2, 3, 4), 0.0, dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'


@pytest.fixture
def cl_file_with_a(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'bcc_csm_1_1_cl_a.nc')
    dataset = Dataset(nc_path, mode='w')
    create_cl_file_without_ap(dataset)
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
    dataset.close()
    return nc_path


@pytest.fixture
def cl_file_with_ap(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'bcc_csm_1_1_cl_ap.nc')
    dataset = Dataset(nc_path, mode='w')
    create_cl_file_without_ap(dataset)
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['ap'][:] = [1.0, 2.0]
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['ap'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'ap: ap b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'ap: ap_bnds b: b_bnds ps: ps')
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


def test_cl_fix_metadata_with_a(cl_file_with_a):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = iris.load(cl_file_with_a)

    # Raw cubes
    assert len(cubes) == 4
    var_names = [cube.var_name for cube in cubes]
    assert 'cl' in var_names
    assert 'ps' in var_names
    assert 'a_bnds' in var_names
    assert 'b_bnds' in var_names

    # Raw cl cube
    cl_cube = cubes.extract_strict('cloud_area_fraction_in_atmosphere_layer')
    air_pressure_coord = cl_cube.coord('air_pressure')
    assert air_pressure_coord.points is not None
    assert air_pressure_coord.bounds is None
    np.testing.assert_allclose(air_pressure_coord.points, AIR_PRESSURE_POINTS)

    # Apply fix
    fix = Cl(None)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    fixed_air_pressure_coord = fixed_cl_cube.coord('air_pressure')
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(fixed_air_pressure_coord.points,
                               AIR_PRESSURE_POINTS)
    np.testing.assert_allclose(fixed_air_pressure_coord.bounds,
                               AIR_PRESSURE_BOUNDS)


def test_cl_fix_metadata_with_ap(cl_file_with_ap):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = iris.load(cl_file_with_ap)

    # Raw cubes
    assert len(cubes) == 4
    var_names = [cube.var_name for cube in cubes]
    assert 'cl' in var_names
    assert 'ps' in var_names
    assert 'ap_bnds' in var_names
    assert 'b_bnds' in var_names

    # Raw cl cube
    cl_cube = cubes.extract_strict('cloud_area_fraction_in_atmosphere_layer')
    air_pressure_coord = cl_cube.coord('air_pressure')
    assert air_pressure_coord.points is not None
    assert air_pressure_coord.bounds is None
    np.testing.assert_allclose(air_pressure_coord.points, AIR_PRESSURE_POINTS)

    # Apply fix
    fix = Cl(None)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    fixed_air_pressure_coord = fixed_cl_cube.coord('air_pressure')
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(fixed_air_pressure_coord.points,
                               AIR_PRESSURE_POINTS)
    np.testing.assert_allclose(fixed_air_pressure_coord.bounds,
                               AIR_PRESSURE_BOUNDS)


class TestTos(unittest.TestCase):
    """Test tos fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'BCC-CSM1-1', 'Amon', 'tos'), [Tos(None)])


def test_tos_fix_data():
    """Test ``fix_data`` for ``tos``."""
    grid_lat = iris.coords.DimCoord(
        [20.0, 40.0],
        bounds=[[10.0, 30.0], [30.0, 50.0]],
        var_name='rlat',
        standard_name='grid_latitude',
    )
    grid_lon = iris.coords.DimCoord(
        [10.0, 20.0, 30.0],
        bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
        var_name='rlon',
        standard_name='grid_longitude',
    )
    latitude = iris.coords.AuxCoord(
        [[-40.0, -20.0, 0.0], [-20.0, 0.0, 20.0]],
        var_name='lat',
        standard_name='latitude',
        units='degrees_north',
    )
    longitude = iris.coords.AuxCoord(
        [[100.0, 140.0, 180.0], [80.0, 100.0, 120.0]],
        var_name='lon',
        standard_name='longitude',
        units='degrees_east',
    )

    # Create cube without bounds
    cube = iris.cube.Cube(
        np.full((2, 3), 300.0),
        var_name='tos',
        units='K',
        dim_coords_and_dims=[(grid_lat, 0), (grid_lon, 1)],
        aux_coords_and_dims=[(latitude, (0, 1)), (longitude, (0, 1))],
    )
    assert cube.coord('latitude').bounds is None
    assert cube.coord('longitude').bounds is None

    # Apply fix
    fix = Tos(None)
    fixed_cube = fix.fix_data(cube)
    assert fixed_cube is cube
    assert fixed_cube.coord('latitude').bounds is not None
    assert fixed_cube.coord('longitude').bounds is not None
    latitude_bounds = np.array([[[-40, -33.75, -23.75, -30.0],
                                 [-33.75, -6.25, 3.75, -23.75],
                                 [-6.25, -1.02418074021670e-14, 10.0, 3.75]],
                                [[-30.0, -23.75, -13.75, -20.0],
                                 [-23.75, 3.75, 13.75, -13.75],
                                 [3.75, 10.0, 20.0, 13.75]]])
    np.testing.assert_allclose(fixed_cube.coord('latitude').bounds,
                               latitude_bounds)
    longitude_bounds = np.array([[[140.625, 99.375, 99.375, 140.625],
                                  [99.375, 140.625, 140.625, 99.375],
                                  [140.625, 99.375, 99.375, 140.625]],
                                 [[140.625, 99.375, 99.375, 140.625],
                                  [99.375, 140.625, 140.625, 99.375],
                                  [140.625, 99.375, 99.375, 140.625]]])
    np.testing.assert_allclose(fixed_cube.coord('longitude').bounds,
                               longitude_bounds)
