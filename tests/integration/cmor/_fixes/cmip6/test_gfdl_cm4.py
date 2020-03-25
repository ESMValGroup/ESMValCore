"""Tests for the fixes of GFDL-CM4."""
import os

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.gfdl_cm4 import Cl, Cli, Clw
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cl_file(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'gfdl_cm4_cl.nc')
    dataset = Dataset(nc_path, mode='w')
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=3)
    dataset.createDimension('lat', size=2)
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
    dataset.variables['lev'][:] = [1.0, 2.0, 4.0]
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].units = '1'
    dataset.variables['lev'].formula_term = (
        'ap: ap b: b ps: ps')  # Error in attribute intended
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0], [3.0, 5.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_term = (
        'ap: ap_bnds b: b_bnds ps: ps')  # Error in attribute intended
    dataset.variables['lat'][:] = [-30.0, 0.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['ap'][:] = [1.0, 2.0, 5.0]
    dataset.variables['ap'].units = 'Pa'
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5], [1.5, 3.0], [3.0, 6.0]]
    dataset.variables['b'][:] = [0.0, 1.0, 3.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0], [2.0, 5.0]]
    dataset.variables['ps'][:] = np.arange(1 * 2 * 2).reshape(1, 2, 2)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'

    # Cl variable
    dataset.createVariable('cl', np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables['cl'][:] = np.full((1, 3, 2, 2), 0.0, dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'

    dataset.close()
    return nc_path


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GFDL-CM4', 'Amon', 'cl')
    assert fix == [Cl(None)]


AIR_PRESSURE_POINTS = np.array([[[[1.0, 1.0],
                                  [1.0, 1.0]],
                                 [[2.0, 3.0],
                                  [4.0, 5.0]],
                                 [[5.0, 8.0],
                                  [11.0, 14.0]]]])
AIR_PRESSURE_BOUNDS = np.array([[[[[0.0, 1.5],
                                   [-1.0, 2.0]],
                                  [[-2.0, 2.5],
                                   [-3.0, 3.0]]],
                                 [[[1.5, 3.0],
                                   [2.0, 5.0]],
                                  [[2.5, 7.0],
                                   [3.0, 9.0]]],
                                 [[[3.0, 6.0],
                                   [5.0, 11.0]],
                                  [[7.0, 16.0],
                                   [9.0, 21.0]]]]])


def test_cl_fix_metadata(cl_file):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = iris.load(cl_file)

    # Raw cubes
    assert len(cubes) == 6
    var_names = [cube.var_name for cube in cubes]
    assert 'cl' in var_names
    assert 'ap' in var_names
    assert 'ap_bnds' in var_names
    assert 'b' in var_names
    assert 'b_bnds' in var_names
    assert 'ps' in var_names

    # Raw cl cube
    cl_cube = cubes.extract_strict('cloud_area_fraction_in_atmosphere_layer')
    assert not cl_cube.coords('air_pressure')

    # Apply fix
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    fix = Cl(vardef)
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


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GFDL-CM4', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GFDL-CM4', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl
