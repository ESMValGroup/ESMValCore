"""Tests for the fixes of CESM2."""
import os
import sys
import unittest.mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.cesm2 import Cl, Cli, Clw, Tas
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cl_file(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'cesm2_cl.nc')
    dataset = Dataset(nc_path, mode='w')
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
    dataset.variables['lev'].units = 'hPa'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')
    dataset.variables['lat'][:] = [-30.0, 0.0, 30.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0, 90.0, 120.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('p0', np.float64, dimensions=())
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a'].bounds = 'a_bnds'
    dataset.variables['a_bnds'][:] = [[1.5, 3.0], [0.0, 1.5]]  # intended
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b'].bounds = 'b_bnds'
    dataset.variables['b_bnds'][:] = [[0.5, 2.0], [-1.0, 0.5]]  # intended
    dataset.variables['p0'][:] = 1.0
    dataset.variables['p0'].units = 'Pa'
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

    dataset.close()
    return nc_path


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2', 'Amon', 'cl')
    assert fix == [Cl(None)]


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


@pytest.mark.skipif(sys.version_info < (3, 7, 6),
                    reason="requires python3.7.6 or newer")
@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.cesm2.Fix.get_fixed_filepath',
    autospec=True)
def test_cl_fix_file(mock_get_filepath, cl_file, tmp_path):
    """Test ``fix_file`` for ``cl``."""
    cubes = iris.load(cl_file)

    # Raw cubes
    assert len(cubes) == 5
    var_names = [cube.var_name for cube in cubes]
    assert 'cl' in var_names
    assert 'a' in var_names
    assert 'b' in var_names
    assert 'p0' in var_names
    assert 'ps' in var_names

    # Raw cl cube
    raw_cube = cubes.extract_strict('cloud_area_fraction_in_atmosphere_layer')
    assert not raw_cube.coords('air_pressure')

    # Apply fix
    mock_get_filepath.return_value = os.path.join(tmp_path,
                                                  'fixed_cesm2_cl.nc')
    fix = Cl(None)
    fixed_file = fix.fix_file(cl_file, tmp_path)
    mock_get_filepath.assert_called_once_with(tmp_path, cl_file)
    fixed_cubes = iris.load(fixed_file)
    assert len(fixed_cubes) == 2
    var_names = [cube.var_name for cube in fixed_cubes]
    assert 'cl' in var_names
    assert 'ps' in var_names
    fixed_cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    fixed_air_pressure_coord = fixed_cl_cube.coord('air_pressure')
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    np.testing.assert_allclose(fixed_air_pressure_coord.points,
                               AIR_PRESSURE_POINTS)
    np.testing.assert_allclose(fixed_air_pressure_coord.bounds,
                               AIR_PRESSURE_BOUNDS)


@pytest.fixture
def cl_cube():
    """``cl`` cube."""
    time_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='time', standard_name='time',
        units='days since 1850-01-01 00:00:00')
    lev_coord = iris.coords.DimCoord(
        [0.0, 1.0, 2.0], var_name='lev',
        standard_name='atmosphere_hybrid_sigma_pressure_coordinate', units='1',
        attributes={'positive': 'up'})
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lat', standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lon', standard_name='longitude', units='degrees')
    coord_specs = [
        (time_coord, 0),
        (lev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    cube = iris.cube.Cube(
        np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2),
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        units='%',
        dim_coords_and_dims=coord_specs,
    )
    return cube


def test_cl_fix_data(cl_cube):
    """Test ``fix_data`` for ``cl``."""
    fix = Cl(None)
    out_cube = fix.fix_data(cl_cube)
    assert out_cube.shape == cl_cube.shape
    np.testing.assert_allclose(out_cube.data,
                               [[[[8, 9],
                                  [10, 11]],
                                 [[4, 5],
                                  [6, 7]],
                                 [[0, 1],
                                  [2, 3]]],
                                [[[20, 21],
                                  [22, 23]],
                                 [[16, 17],
                                  [18, 19]],
                                 [[12, 13],
                                  [14, 15]]]])
    np.testing.assert_allclose(out_cube.coord(var_name='lev').points,
                               [2.0, 1.0, 0.0])


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


@pytest.fixture
def tas_cubes():
    """Cubes to test fixes for ``tas``."""
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CESM2', 'Amon', 'tas')
    assert fix == [Tas(None)]


def test_tas_fix_metadata(tas_cubes):
    """Test ``fix_metadata`` for ``tas``."""
    for cube in tas_cubes:
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
    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes is tas_cubes
    for cube in out_cubes:
        if cube.var_name == 'tas':
            coord = cube.coord('height')
            assert coord == height_coord
        else:
            with pytest.raises(iris.exceptions.CoordinateNotFoundError):
                cube.coord('height')
