"""Test for common fixes used for multiple datasets."""
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint

from esmvalcore.cmor._fixes.common import (
    ClFixHybridHeightCoord,
    ClFixHybridPressureCoord,
    OceanFixGrid,
    SiconcFixScalarCoord,
)
from esmvalcore.cmor.table import get_var_info

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
    cube = cubes.extract_cube(NameConstraint(var_name=short_name))
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
    fixed_cube = fixed_cubes.extract_cube(NameConstraint(var_name=short_name))
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


@pytest.mark.sequential
def test_cl_hybrid_pressure_coord_fix_metadata_with_a(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    nc_path = test_data_path / 'common_cl_a.nc'
    var_names = hybrid_pressure_coord_fix_metadata(
        nc_path, 'cl', ClFixHybridPressureCoord(vardef))
    assert 'a_bnds' in var_names


@pytest.mark.sequential
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
    cube = cubes.extract_cube(NameConstraint(var_name=short_name))
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
    fixed_cube = fixed_cubes.extract_cube(NameConstraint(var_name=short_name))
    fixed_height_coord = fixed_cube.coord('altitude')
    assert fixed_height_coord.points is not None
    assert fixed_height_coord.bounds is not None
    np.testing.assert_allclose(fixed_height_coord.points, HEIGHT_POINTS)
    np.testing.assert_allclose(fixed_height_coord.bounds, HEIGHT_BOUNDS_RIGHT)
    assert not np.allclose(fixed_height_coord.bounds, HEIGHT_BOUNDS_WRONG)
    air_pressure_coord = cube.coord('air_pressure')
    np.testing.assert_allclose(air_pressure_coord.points, PRESSURE_POINTS)
    np.testing.assert_allclose(air_pressure_coord.bounds, PRESSURE_BOUNDS)
    assert air_pressure_coord.var_name is None
    assert air_pressure_coord.standard_name == 'air_pressure'
    assert air_pressure_coord.long_name is None
    assert air_pressure_coord.units == 'Pa'


@pytest.mark.sequential
def test_cl_hybrid_height_coord_fix_metadata(test_data_path):
    """Test ``fix_metadata`` for ``cl``."""
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    nc_path = test_data_path / 'common_cl_hybrid_height.nc'
    hybrid_height_coord_fix_metadata(nc_path, 'cl',
                                     ClFixHybridHeightCoord(vardef))


@pytest.fixture
def siconc_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.0], standard_name='time',
                                      var_name='time',
                                      units='days since 6543-2-1')
    lat_coord = iris.coords.DimCoord([-30.0], standard_name='latitude',
                                     var_name='lat', units='degrees_north')
    lon_coord = iris.coords.DimCoord([30.0], standard_name='longitude',
                                     var_name='lon', units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0]]], standard_name='sea_ice_area_fraction',
                          var_name='siconc', units='%',
                          dim_coords_and_dims=coords_specs)
    return iris.cube.CubeList([cube])


def test_siconc_fix_metadata(siconc_cubes):
    """Test ``fix_metadata`` for ``siconc``."""
    assert len(siconc_cubes) == 1
    siconc_cube = siconc_cubes[0]
    assert siconc_cube.var_name == "siconc"

    # Extract siconc cube
    siconc_cube = siconc_cubes.extract_cube('sea_ice_area_fraction')
    assert not siconc_cube.coords('typesi')

    # Apply fix
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = SiconcFixScalarCoord(vardef)
    fixed_cubes = fix.fix_metadata(siconc_cubes)
    assert len(fixed_cubes) == 1
    fixed_siconc_cube = fixed_cubes.extract_cube(
        'sea_ice_area_fraction')
    fixed_typesi_coord = fixed_siconc_cube.coord('area_type')
    assert fixed_typesi_coord.points is not None
    assert fixed_typesi_coord.bounds is None
    np.testing.assert_equal(fixed_typesi_coord.points,
                            ['sea_ice'])
    np.testing.assert_equal(fixed_typesi_coord.units,
                            Unit('No unit'))


def get_tos_cubes(wrong_ij_names=False, ij_bounds=False):
    """Cubes containing tos variable."""
    if wrong_ij_names:
        j_var_name = 'lat'
        j_long_name = 'latitude'
        i_var_name = 'lon'
        i_long_name = 'longitude'
    else:
        j_var_name = 'j'
        j_long_name = 'cell index along second dimension'
        i_var_name = 'i'
        i_long_name = 'cell index along first dimension'
    if ij_bounds:
        j_bounds = [[10.0, 30.0], [30.0, 50.0]]
        i_bounds = [[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]]
    else:
        j_bounds = None
        i_bounds = None
    j_coord = iris.coords.DimCoord(
        [20.0, 40.0],
        bounds=j_bounds,
        var_name=j_var_name,
        long_name=j_long_name,
    )
    i_coord = iris.coords.DimCoord(
        [10.0, 20.0, 30.0],
        bounds=i_bounds,
        var_name=i_var_name,
        long_name=i_long_name,
    )
    lat_coord = iris.coords.AuxCoord(
        [[-40.0, -20.0, 0.0], [-20.0, 0.0, 20.0]],
        var_name='lat',
        standard_name='latitude',
        units='degrees_north',
    )
    lon_coord = iris.coords.AuxCoord(
        [[100.0, 140.0, 180.0], [80.0, 100.0, 120.0]],
        var_name='lon',
        standard_name='longitude',
        units='degrees_east',
    )
    time_coord = iris.coords.DimCoord(
        1.0,
        bounds=[0.0, 2.0],
        var_name='time',
        standard_name='time',
        long_name='time',
        units='days since 1950-01-01',
    )

    # Create tos variable cube
    cube = iris.cube.Cube(
        np.full((1, 2, 3), 300.0),
        var_name='tos',
        long_name='sea_surface_temperature',
        units='K',
        dim_coords_and_dims=[(time_coord, 0), (j_coord, 1), (i_coord, 2)],
        aux_coords_and_dims=[(lat_coord, (1, 2)), (lon_coord, (1, 2))],
    )

    # Create empty (dummy) cube
    empty_cube = iris.cube.Cube(0.0)
    return iris.cube.CubeList([cube, empty_cube])


@pytest.fixture
def tos_cubes_wrong_ij_names():
    """Cubes with wrong ij names."""
    return get_tos_cubes(wrong_ij_names=True, ij_bounds=True)


def test_ocean_fix_grid_wrong_ij_names(tos_cubes_wrong_ij_names):
    """Test ``fix_metadata`` with cubes with wrong ij names."""
    cube_in = tos_cubes_wrong_ij_names.extract_cube('sea_surface_temperature')
    assert len(cube_in.coords('latitude')) == 2
    assert len(cube_in.coords('longitude')) == 2
    assert cube_in.coord('latitude', dimensions=1).bounds is not None
    assert cube_in.coord('longitude', dimensions=2).bounds is not None
    assert cube_in.coord('latitude', dimensions=(1, 2)).bounds is None
    assert cube_in.coord('longitude', dimensions=(1, 2)).bounds is None

    # Apply fix
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = OceanFixGrid(vardef)
    fixed_cubes = fix.fix_metadata(tos_cubes_wrong_ij_names)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_cube('sea_surface_temperature')
    assert fixed_cube is cube_in

    # Check ij names
    i_coord = fixed_cube.coord('cell index along first dimension')
    j_coord = fixed_cube.coord('cell index along second dimension')
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == 'cell index along first dimension'
    assert i_coord.units == '1'
    assert i_coord.circular is False
    assert j_coord.var_name == 'j'
    assert j_coord.standard_name is None
    assert j_coord.long_name == 'cell index along second dimension'
    assert j_coord.units == '1'

    # Check ij points and bounds
    np.testing.assert_allclose(i_coord.points, [0, 1, 2])
    np.testing.assert_allclose(i_coord.bounds,
                               [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])
    np.testing.assert_allclose(j_coord.points, [0, 1])
    np.testing.assert_allclose(j_coord.bounds, [[-0.5, 0.5], [0.5, 1.5]])

    # Check bounds of latitude and longitude
    assert len(fixed_cube.coords('latitude')) == 1
    assert len(fixed_cube.coords('longitude')) == 1
    assert fixed_cube.coord('latitude').bounds is not None
    assert fixed_cube.coord('longitude').bounds is not None
    latitude_bounds = np.array(
        [[[-43.48076211, -34.01923789, -22.00961894, -31.47114317],
          [-34.01923789, -10.0, 2.00961894, -22.00961894],
          [-10.0, -0.53847577, 11.47114317, 2.00961894]],
         [[-31.47114317, -22.00961894, -10.0, -19.46152423],
          [-22.00961894, 2.00961894, 14.01923789, -10.0],
          [2.00961894, 11.47114317, 23.48076211, 14.01923789]]]
    )
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


@pytest.fixture
def tos_cubes_no_ij_bounds():
    """Cubes with no ij bounds."""
    return get_tos_cubes(wrong_ij_names=False, ij_bounds=False)


def test_ocean_fix_grid_no_ij_bounds(tos_cubes_no_ij_bounds):
    """Test ``fix_metadata`` with cubes with no ij bounds."""
    cube_in = tos_cubes_no_ij_bounds.extract_cube('sea_surface_temperature')
    assert len(cube_in.coords('latitude')) == 1
    assert len(cube_in.coords('longitude')) == 1
    assert cube_in.coord('latitude').bounds is None
    assert cube_in.coord('longitude').bounds is None
    assert cube_in.coord('cell index along first dimension').var_name == 'i'
    assert cube_in.coord('cell index along second dimension').var_name == 'j'
    assert cube_in.coord('cell index along first dimension').bounds is None
    assert cube_in.coord('cell index along second dimension').bounds is None

    # Apply fix
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = OceanFixGrid(vardef)
    fixed_cubes = fix.fix_metadata(tos_cubes_no_ij_bounds)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_cube('sea_surface_temperature')
    assert fixed_cube is cube_in

    # Check ij names
    i_coord = fixed_cube.coord('cell index along first dimension')
    j_coord = fixed_cube.coord('cell index along second dimension')
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == 'cell index along first dimension'
    assert i_coord.units == '1'
    assert i_coord.circular is False
    assert j_coord.var_name == 'j'
    assert j_coord.standard_name is None
    assert j_coord.long_name == 'cell index along second dimension'
    assert j_coord.units == '1'

    # Check ij points and bounds
    np.testing.assert_allclose(i_coord.points, [0, 1, 2])
    np.testing.assert_allclose(i_coord.bounds,
                               [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])
    np.testing.assert_allclose(j_coord.points, [0, 1])
    np.testing.assert_allclose(j_coord.bounds, [[-0.5, 0.5], [0.5, 1.5]])

    # Check bounds of latitude and longitude
    assert len(fixed_cube.coords('latitude')) == 1
    assert len(fixed_cube.coords('longitude')) == 1
    assert fixed_cube.coord('latitude').bounds is not None
    assert fixed_cube.coord('longitude').bounds is not None
    latitude_bounds = np.array(
        [[[-43.48076211, -34.01923789, -22.00961894, -31.47114317],
          [-34.01923789, -10.0, 2.00961894, -22.00961894],
          [-10.0, -0.53847577, 11.47114317, 2.00961894]],
         [[-31.47114317, -22.00961894, -10.0, -19.46152423],
          [-22.00961894, 2.00961894, 14.01923789, -10.0],
          [2.00961894, 11.47114317, 23.48076211, 14.01923789]]]
    )
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
