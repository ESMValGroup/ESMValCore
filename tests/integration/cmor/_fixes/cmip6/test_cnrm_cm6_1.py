"""Tests for the fixes of CNRM-CM6-1."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.cnrm_cm6_1 import (Cl, Clcalipso,
                                                     Cli, Clw, Omon)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1', 'Amon', 'cl')
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


def test_cl_fix_metadata(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    nc_path = test_data_path / 'cnrm_cm6_1_cl.nc'
    cubes = iris.load(str(nc_path))

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
    cl_cube = cubes.extract_cube('cloud_area_fraction_in_atmosphere_layer')
    assert not cl_cube.coords('air_pressure')

    # Apply fix
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    fix = Cl(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cl_cube = fixed_cubes.extract_cube(
        'cloud_area_fraction_in_atmosphere_layer')
    fixed_air_pressure_coord = fixed_cl_cube.coord('air_pressure')
    assert fixed_air_pressure_coord.points is not None
    assert fixed_air_pressure_coord.bounds is not None
    assert fixed_air_pressure_coord.points.shape == (1, 3, 2, 2)
    assert fixed_air_pressure_coord.bounds.shape == (1, 3, 2, 2, 2)
    np.testing.assert_allclose(fixed_air_pressure_coord.points,
                               AIR_PRESSURE_POINTS)
    np.testing.assert_allclose(fixed_air_pressure_coord.bounds,
                               AIR_PRESSURE_BOUNDS)
    lat_coord = fixed_cl_cube.coord('latitude')
    lon_coord = fixed_cl_cube.coord('longitude')
    assert lat_coord.bounds is not None
    assert lon_coord.bounds is not None
    np.testing.assert_allclose(lat_coord.bounds,
                               [[-45.0, -15.0], [-15.0, 15.0]])
    np.testing.assert_allclose(lon_coord.bounds,
                               [[15.0, 45.0], [45.0, 75.0]])


def test_get_clcalipso_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1', 'CFmon', 'clcalipso')
    assert fix == [Clcalipso(None)]


@pytest.fixture
def clcalipso_cubes():
    """Cubes to test fix for ``clcalipso``."""
    alt_40_coord = iris.coords.DimCoord([0.0], var_name='alt40')
    cube = iris.cube.Cube([0.0], var_name='clcalipso',
                          dim_coords_and_dims=[(alt_40_coord.copy(), 0)])
    x_cube = iris.cube.Cube([0.0], var_name='x',
                            dim_coords_and_dims=[(alt_40_coord.copy(), 0)])
    return iris.cube.CubeList([cube, x_cube])


def test_clcalipso_fix_metadata(clcalipso_cubes):
    """Test ``fix_metadata`` for ``clcalipso``."""
    vardef = get_var_info('CMIP6', 'CFmon', 'clcalipso')
    fix = Clcalipso(vardef)
    cubes = fix.fix_metadata(clcalipso_cubes)
    assert len(cubes) == 1
    cube = cubes[0]
    coord = cube.coord('altitude')
    assert coord.standard_name == 'altitude'


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CNRM-CM6-1', 'Omon', 'thetao')
    assert fix == [Omon(None)]
