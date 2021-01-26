"""Test for common fixes used for multiple datasets."""
import iris
import numpy as np

from esmvalcore.cmor._fixes.common import (
    ClFixHybridHeightCoord,
    ClFixHybridPressureCoord,
)
from esmvalcore.cmor.table import get_var_info
from esmvalcore.iris_helpers import var_name_constraint


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
    cubes = iris.load(str(nc_path))

    # Raw cubes
    assert len(cubes) == 4
    var_names = [cube.var_name for cube in cubes]
    assert short_name in var_names
    assert 'ps' in var_names
    assert 'b_bnds' in var_names

    # Raw cube
    cube = cubes.extract_cube(var_name_constraint(short_name))
    air_pressure_coord = cube.coord('air_pressure')
    assert air_pressure_coord.points is not None
    assert air_pressure_coord.bounds is None
    np.testing.assert_allclose(air_pressure_coord.points, AIR_PRESSURE_POINTS)

    # Raw ps cube
    ps_cube = cubes.extract_cube('surface_air_pressure')
    assert ps_cube.attributes == {'additional_attribute': 'xyz'}

    # Apply fix
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_cube(var_name_constraint(short_name))
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


def test_cl_hybrid_pressure_coord_fix_metadata_with_a(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    nc_path = test_data_path / 'common_cl_a.nc'
    var_names = hybrid_pressure_coord_fix_metadata(
        nc_path, 'cl', ClFixHybridPressureCoord(vardef))
    assert 'a_bnds' in var_names


def test_cl_hybrid_pressure_coord_fix_metadata_with_ap(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    nc_path = test_data_path / 'common_cl_ap.nc'
    var_names = hybrid_pressure_coord_fix_metadata(
        nc_path, 'cl', ClFixHybridPressureCoord(vardef))
    assert 'ap_bnds' in var_names


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
    cubes = iris.load(str(nc_path))

    # Raw cubes
    assert len(cubes) == 3
    var_names = [cube.var_name for cube in cubes]
    assert short_name in var_names
    assert 'orog' in var_names
    assert 'b_bnds' in var_names

    # Raw cube
    cube = cubes.extract_cube(var_name_constraint(short_name))
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
    fixed_cube = fixed_cubes.extract_cube(var_name_constraint(short_name))
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


def test_cl_hybrid_height_coord_fix_metadata(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    nc_path = test_data_path / 'common_cl_hybrid_height.nc'
    hybrid_height_coord_fix_metadata(nc_path, 'cl',
                                     ClFixHybridHeightCoord(vardef))
