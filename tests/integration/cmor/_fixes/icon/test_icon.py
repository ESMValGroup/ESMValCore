"""Tests for the ICON on-the-fly CMORizer."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore._config import get_extra_facets
from esmvalcore.cmor._fixes.icon.icon import AllVars, Siconca
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / 'icon_2d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_3d(test_data_path):
    """3D sample cubes."""
    nc_path = test_data_path / 'icon_3d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_grid(test_data_path):
    """Grid description sample cubes."""
    nc_path = test_data_path / 'icon_grid.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_regular_grid():
    """Cube with regular grid."""
    lat_coord = iris.coords.DimCoord([0.0, 1.0], var_name='lat',
                                     standard_name='latitude',
                                     long_name='latitude',
                                     units='degrees_north')
    lon_coord = iris.coords.DimCoord([-1.0, 1.0], var_name='lon',
                                     standard_name='longitude',
                                     long_name='longitude',
                                     units='degrees_east')
    cube = iris.cube.Cube([[0.0, 1.0], [2.0, 3.0]], var_name='tas', units='K',
                          dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)])
    return iris.cube.CubeList([cube])


@pytest.fixture
def cubes_2d_lat_lon_grid():
    """Cube with 2D latitude and longitude."""
    lat_coord = iris.coords.AuxCoord([[0.0, 0.0], [1.0, 1.0]], var_name='lat',
                                     standard_name='latitude',
                                     long_name='latitude',
                                     units='degrees_north')
    lon_coord = iris.coords.AuxCoord([[0.0, 1.0], [0.0, 1.0]], var_name='lon',
                                     standard_name='longitude',
                                     long_name='longitude',
                                     units='degrees_east')
    cube = iris.cube.Cube([[0.0, 1.0], [2.0, 3.0]], var_name='tas', units='K',
                          aux_coords_and_dims=[(lat_coord, (0, 1)),
                                               (lon_coord, (0, 1))])
    return iris.cube.CubeList([cube])


def get_allvars_fix(mip, short_name):
    """Get member of fix class."""
    vardef = get_var_info('ICON', mip, short_name)
    extra_facets = get_extra_facets('ICON', 'ICON', mip, short_name, ())
    fix = AllVars(vardef, extra_facets=extra_facets)
    return fix


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('days since 1850-01-01',
                              calendar='proleptic_gregorian')
    np.testing.assert_allclose(time.points, [54786.])
    assert time.bounds is None
    assert time.attributes == {}


def check_height(cube):
    """Check height coordinate of cube."""
    assert cube.coords('model level number', dim_coords=True)
    height = cube.coord('model level number', dim_coords=True)
    assert height.var_name == 'model_level'
    assert height.standard_name is None
    assert height.long_name == 'model level number'
    assert height.units == 'no unit'
    np.testing.assert_array_equal(height.points, np.arange(47))
    assert height.bounds is None
    assert height.attributes == {'positive': 'up'}

    assert cube.coords('air_pressure', dim_coords=False)
    plev = cube.coord('air_pressure', dim_coords=False)
    assert plev.var_name == 'plev'
    assert plev.standard_name == 'air_pressure'
    assert plev.long_name == 'pressure'
    assert plev.units == 'Pa'
    assert plev.bounds is not None
    assert plev.attributes == {'positive': 'down'}
    assert cube.coord_dims('air_pressure') == (0, 1, 2)


def check_lat_lon(cube):
    """Check latitude and longitude coordinates of cube."""
    assert cube.coords('latitude', dim_coords=False)
    lat = cube.coord('latitude', dim_coords=False)
    assert lat.var_name == 'lat'
    assert lat.standard_name == 'latitude'
    assert lat.long_name == 'latitude'
    assert lat.units == 'degrees_north'
    np.testing.assert_allclose(
        lat.points,
        [-45.0, -45.0, -45.0, -45.0, 45.0, 45.0, 45.0, 45.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        lat.bounds,
        [
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
        ],
        rtol=1e-5
    )

    assert cube.coords('longitude', dim_coords=False)
    lon = cube.coord('longitude', dim_coords=False)
    assert lon.var_name == 'lon'
    assert lon.standard_name == 'longitude'
    assert lon.long_name == 'longitude'
    assert lon.units == 'degrees_east'
    np.testing.assert_allclose(
        lon.points,
        [-135.0, -45.0, 45.0, 135.0, -135.0, -45.0, 45.0, 135.0],
        rtol=1e-5
    )
    np.testing.assert_allclose(
        lon.bounds,
        [
            [-135.0, -90.0, -180.0],
            [-45.0, 0.0, -90.0],
            [45.0, 90.0, 0.0],
            [135.0, 180.0, 90.0],
            [-180.0, -90.0, -135.0],
            [-90.0, 0.0, -45.0],
            [0.0, 90.0, 45.0],
            [90.0, 180.0, 135.0],
        ],
        rtol=1e-5
    )

    assert cube.coords('first spatial index for variables stored on an '
                       'unstructured grid', dim_coords=True)
    i_coord = cube.coord('first spatial index for variables stored on an '
                         'unstructured grid', dim_coords=True)
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == ('first spatial index for variables stored on '
                                 'an unstructured grid')
    assert i_coord.units == '1'
    np.testing.assert_allclose(i_coord.points, [0, 1, 2, 3, 4, 5, 6, 7])
    assert i_coord.bounds is None

    assert len(cube.coord_dims(lat)) == 1
    assert cube.coord_dims(lat) == cube.coord_dims(lon)
    assert cube.coord_dims(lat) == cube.coord_dims(i_coord)


# Test areacella (for extra_facets, and grid_latitude and grid_longitude
# coordinates)


def test_get_areacella_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'fx', 'areacella')
    assert fix == [AllVars(None)]


def test_areacella_fix(cubes_grid):
    """Test fix."""
    fix = get_allvars_fix('fx', 'areacella')
    fixed_cubes = fix.fix_metadata(cubes_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'areacella'
    assert cube.standard_name == 'cell_area'
    assert cube.long_name == 'Grid-Cell Area for Atmospheric Grid Variables'
    assert cube.units == 'm2'

    check_lat_lon(cube)


# Test clwvi (for extra_facets)


def test_get_clwvi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'clwvi')
    assert fix == [AllVars(None)]


def test_clwvi_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'clwvi')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clwvi'
    assert cube.standard_name == ('atmosphere_mass_content_of_cloud_'
                                  'condensed_water')
    assert cube.long_name == 'Condensed Water Path'
    assert cube.units == 'kg m-2'

    check_time(cube)
    check_lat_lon(cube)


# Test siconca (for extra_facets, extra fix and typesi coordinate)


def test_get_siconca_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'SImon', 'siconca')
    assert fix == [Siconca(None), AllVars(None)]


def test_siconca_fix(cubes_2d):
    """Test fix."""
    vardef = get_var_info('ICON', 'SImon', 'siconca')
    extra_facets = get_extra_facets('ICON', 'ICON', 'SImon', 'siconca', ())
    siconca_fix = Siconca(vardef, extra_facets=extra_facets)
    allvars_fix = get_allvars_fix('SImon', 'siconca')

    fixed_cubes = siconca_fix.fix_metadata(cubes_2d)
    fixed_cubes = allvars_fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'siconca'
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == 'Sea-Ice Area Percentage (Atmospheric Grid)'
    assert cube.units == '%'

    check_time(cube)
    check_lat_lon(cube)
    assert cube.coords('area_type')
    typesi = cube.coord('area_type')
    assert typesi.var_name == 'type'
    assert typesi.standard_name == 'area_type'
    assert typesi.long_name == 'Sea Ice area type'
    assert typesi.units.is_no_unit()
    np.testing.assert_array_equal(typesi.points, ['sea_ice'])
    assert typesi.bounds is None

    np.testing.assert_allclose(
        cube.data,
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
    )


# Test ta (for height and plev coordinate)


def test_get_ta_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'ta')
    assert fix == [AllVars(None)]


def test_ta_fix(cubes_3d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ta'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Air Temperature'
    assert cube.units == 'K'

    check_time(cube)
    check_height(cube)
    check_lat_lon(cube)


# Test tas (for height2m coordinate)


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_tas_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'

    check_time(cube)
    check_lat_lon(cube)
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [2.0])
    assert height.bounds is None


# Test uas (for height10m coordinate)


def test_get_uas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'uas')
    assert fix == [AllVars(None)]


def test_uas_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'uas'
    assert cube.standard_name == 'eastward_wind'
    assert cube.long_name == 'Eastward Near-Surface Wind'
    assert cube.units == 'm s-1'

    check_time(cube)
    check_lat_lon(cube)
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [10.0])
    assert height.bounds is None


# Test fix with regular grid and 2D latitudes and longitude


def test_regular_grid_fix(cubes_regular_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_regular_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'

    assert cube.coords('latitude', dim_coords=True, dimensions=0)
    assert cube.coords('longitude', dim_coords=True, dimensions=1)
    assert cube.coords('height', dim_coords=False, dimensions=())


def test_2d_lat_lon_grid_fix(cubes_2d_lat_lon_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d_lat_lon_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'

    assert cube.coords('latitude', dim_coords=False, dimensions=(0, 1))
    assert cube.coords('longitude', dim_coords=False, dimensions=(0, 1))
    assert cube.coords('height', dim_coords=False, dimensions=())


# Test fix with empty standard_name


def test_empty_standard_name_fix(cubes_2d):
    """Test fix."""
    # We know that tas has a standard name, but this being native model output
    # there may be variables with no standard name. The code is designed to
    # handle this gracefully and here we test it with an artificial, but
    # realistic case.
    vardef = get_var_info('ICON', 'Amon', 'tas')
    vardef.standard_name = ''
    extra_facets = get_extra_facets('ICON', 'ICON', 'Amon', 'tas', ())
    fix = AllVars(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name is None
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'


# Test fix with invalid time units


def test_invalid_time_units(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    for cube in cubes_2d:
        cube.coord('time').attributes['invalid_units'] = 'month as %Y%m%d.%f'
    with pytest.raises(ValueError):
        fix.fix_metadata(cubes_2d)
