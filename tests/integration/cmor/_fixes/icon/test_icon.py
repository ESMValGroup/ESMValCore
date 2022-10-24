"""Tests for the ICON on-the-fly CMORizer."""
from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore._config import get_extra_facets
from esmvalcore.cmor._fixes.icon.icon import AllVars, Siconc, Siconca
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info

# Note: test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


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
    time_coord = DimCoord([0], var_name='time', standard_name='time',
                          units='days since 1850-01-01')
    lat_coord = DimCoord([0.0, 1.0], var_name='lat', standard_name='latitude',
                         long_name='latitude', units='degrees_north')
    lon_coord = DimCoord([-1.0, 1.0], var_name='lon',
                         standard_name='longitude', long_name='longitude',
                         units='degrees_east')
    cube = Cube([[[0.0, 1.0], [2.0, 3.0]]], var_name='tas', units='K',
                dim_coords_and_dims=[(time_coord, 0),
                                     (lat_coord, 1),
                                     (lon_coord, 2)])
    return CubeList([cube])


@pytest.fixture
def cubes_2d_lat_lon_grid():
    """Cube with 2D latitude and longitude."""
    time_coord = DimCoord([0], var_name='time', standard_name='time',
                          units='days since 1850-01-01')
    lat_coord = AuxCoord([[0.0, 0.0], [1.0, 1.0]], var_name='lat',
                         standard_name='latitude', long_name='latitude',
                         units='degrees_north')
    lon_coord = AuxCoord([[0.0, 1.0], [0.0, 1.0]], var_name='lon',
                         standard_name='longitude', long_name='longitude',
                         units='degrees_east')
    cube = Cube([[[0.0, 1.0], [2.0, 3.0]]], var_name='tas', units='K',
                dim_coords_and_dims=[(time_coord, 0)],
                aux_coords_and_dims=[(lat_coord, (1, 2)),
                                     (lon_coord, (1, 2))])
    return CubeList([cube])


def get_allvars_fix(mip, short_name):
    """Get member of fix class."""
    vardef = get_var_info('ICON', mip, short_name)
    extra_facets = get_extra_facets('ICON', 'ICON', mip, short_name, ())
    fix = AllVars(vardef, extra_facets=extra_facets)
    return fix


def check_ta_metadata(cubes):
    """Check ta metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'ta'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes
    return cube


def check_tas_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes
    return cube


def check_siconc_metadata(cubes, var_name, long_name):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == var_name
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == long_name
    assert cube.units == '%'
    assert 'positive' not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('days since 1850-01-01',
                              calendar='proleptic_gregorian')
    np.testing.assert_allclose(time.points, [54786.0])
    assert time.bounds is None
    assert time.attributes == {}


def check_height(cube, plev_has_bounds=True):
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
    assert plev.attributes == {'positive': 'down'}
    assert cube.coord_dims('air_pressure') == (0, 1, 2)

    np.testing.assert_allclose(
        plev.points[0, :4, 0],
        [100566.234, 99652.07, 97995.77, 95686.08],
    )
    if plev_has_bounds:
        np.testing.assert_allclose(
            plev.bounds[0, :4, 0],
            [[100825.04, 100308.09],
             [100308.09, 99000.336],
             [99000.336, 97001.42],
             [97001.42, 94388.59]],
        )
    else:
        assert plev.bounds is None


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def check_lat(cube):
    """Check latitude coordinate of cube."""
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
    return lat


def check_lon(cube):
    """Check longitude coordinate of cube."""
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
    return lon


def check_lat_lon(cube):
    """Check latitude, longitude and spatial index coordinates of cube."""
    lat = check_lat(cube)
    lon = check_lon(cube)

    # Check spatial index coordinate
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


def check_typesi(cube):
    """Check scalar typesi coordinate of cube."""
    assert cube.coords('area_type')
    typesi = cube.coord('area_type')
    assert typesi.var_name == 'type'
    assert typesi.standard_name == 'area_type'
    assert typesi.long_name == 'Sea Ice area type'
    assert typesi.units.is_no_unit()
    np.testing.assert_array_equal(typesi.points, ['sea_ice'])
    assert typesi.bounds is None


# Test areacella and areacello (for extra_facets, and grid_latitude and
# grid_longitude coordinates)


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
    assert 'positive' not in cube.attributes

    check_lat_lon(cube)


def test_get_areacello_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Ofx', 'areacello')
    assert fix == [AllVars(None)]


def test_areacello_fix(cubes_grid):
    """Test fix."""
    fix = get_allvars_fix('Ofx', 'areacello')
    fixed_cubes = fix.fix_metadata(cubes_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'areacello'
    assert cube.standard_name == 'cell_area'
    assert cube.long_name == 'Grid-Cell Area for Ocean Variables'
    assert cube.units == 'm2'
    assert 'positive' not in cube.attributes

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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat_lon(cube)


# Test rsdt and rsut (for positive attribute)


def test_get_rsdt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rsdt')
    assert fix == [AllVars(None)]


def test_rsdt_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsdt')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsdt'
    assert cube.standard_name == 'toa_incoming_shortwave_flux'
    assert cube.long_name == 'TOA Incident Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat_lon(cube)


def test_get_rsut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'Amon', 'rsut')
    assert fix == [AllVars(None)]


def test_rsut_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsut')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsut'
    assert cube.standard_name == 'toa_outgoing_shortwave_flux'
    assert cube.long_name == 'TOA Outgoing Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat_lon(cube)


# Test siconc and siconca (for extra_facets, extra fix and typesi coordinate)


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('ICON', 'ICON', 'SImon', 'siconc')
    assert fix == [Siconc(None), AllVars(None)]


def test_siconc_fix(cubes_2d):
    """Test fix."""
    vardef = get_var_info('ICON', 'SImon', 'siconc')
    extra_facets = get_extra_facets('ICON', 'ICON', 'SImon', 'siconc', ())
    siconc_fix = Siconc(vardef, extra_facets=extra_facets)
    allvars_fix = get_allvars_fix('SImon', 'siconc')

    fixed_cubes = siconc_fix.fix_metadata(cubes_2d)
    fixed_cubes = allvars_fix.fix_metadata(fixed_cubes)

    cube = check_siconc_metadata(fixed_cubes, 'siconc',
                                 'Sea-Ice Area Percentage (Ocean Grid)')
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data,
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
    )


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

    cube = check_siconc_metadata(fixed_cubes, 'siconca',
                                 'Sea-Ice Area Percentage (Atmospheric Grid)')
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

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

    cube = check_ta_metadata(fixed_cubes)
    check_time(cube)
    check_height(cube)
    check_lat_lon(cube)


def test_ta_fix_no_plev_bounds(cubes_3d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cubes = CubeList([
        cubes_3d.extract_cube(NameConstraint(var_name='ta')),
        cubes_3d.extract_cube(NameConstraint(var_name='pfull')),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_ta_metadata(fixed_cubes)
    check_time(cube)
    check_height(cube, plev_has_bounds=False)
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

    cube = check_tas_metadata(fixed_cubes)
    check_time(cube)
    check_lat_lon(cube)
    check_heightxm(cube, 2.0)


def test_tas_spatial_index_coord_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    index_coord = DimCoord(np.arange(8), var_name='ncells')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_dim_coord(index_coord, 1)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    check_lat_lon(cube)


def test_tas_scalar_height2m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # Scalar height (with wrong metadata) already present
    height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_aux_coord(height_coord, ())
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 2.0)


def test_tas_dim_height2m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # Dimensional coordinate height (with wrong metadata) already present
    height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    cube.add_aux_coord(height_coord, ())
    cube = iris.util.new_axis(cube, scalar_coord='height')
    cube.transpose((1, 0, 2))
    cubes = CubeList([cube])
    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 2.0)


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
    assert 'positive' not in cube.attributes

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


def test_uas_scalar_height10m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')

    # Scalar height (with wrong metadata) already present
    height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    cube.add_aux_coord(height_coord, ())
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 10.0)


def test_uas_dim_height10m_already_present(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')

    # Dimensional coordinate height (with wrong metadata) already present
    height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
    cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    cube.add_aux_coord(height_coord, ())
    cube = iris.util.new_axis(cube, scalar_coord='height')
    cube.transpose((1, 0, 2))
    cubes = CubeList([cube])
    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.shape == (1, 8)
    check_heightxm(cube, 10.0)


# Test fix with regular grid and 2D latitudes and longitude


def test_regular_grid_fix(cubes_regular_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_regular_grid)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.coords('time', dim_coords=True, dimensions=0)
    assert cube.coords('latitude', dim_coords=True, dimensions=1)
    assert cube.coords('longitude', dim_coords=True, dimensions=2)
    assert cube.coords('height', dim_coords=False, dimensions=())


def test_2d_lat_lon_grid_fix(cubes_2d_lat_lon_grid):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d_lat_lon_grid)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.coords('time', dim_coords=True, dimensions=0)
    assert cube.coords('latitude', dim_coords=False, dimensions=(1, 2))
    assert cube.coords('longitude', dim_coords=False, dimensions=(1, 2))
    assert cube.coords('height', dim_coords=False, dimensions=())


# Test fix with empty standard_name


def test_empty_standard_name_fix(cubes_2d):
    """Test fix."""
    # We know that tas has a standard name, but this being native model output
    # there may be variables with no standard name. The code is designed to
    # handle this gracefully and here we test it with an artificial, but
    # realistic case.
    vardef = get_var_info('ICON', 'Amon', 'tas')
    original_standard_name = vardef.standard_name
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
    assert 'positive' not in cube.attributes

    # Restore original standard_name of tas
    vardef.standard_name = original_standard_name


# Test automatic addition of missing coordinates


def test_add_time(cubes_2d):
    """Test fix."""
    # Remove time from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    uas_cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
    tas_cube = tas_cube[0]
    tas_cube.remove_coord('time')
    cubes = CubeList([tas_cube, uas_cube])

    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_time(cube)


def test_add_time_fail():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cube = Cube(1, var_name='ta', units='K')
    cubes = CubeList([
        cube,
        Cube(1, var_name='tas', units='K'),
    ])
    msg = "Cannot add required coordinate 'time' to variable 'ta'"
    with pytest.raises(ValueError, match=msg):
        fix._add_time(cube, cubes)


def test_add_latitude(cubes_2d, tmp_path):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.attributes['grid_file_uri'] = (
        'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/'
        'integration/cmor/_fixes/test_data/icon_grid.nc'
    )
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert 'icon_grid.nc' in fix._horizontal_grids

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


def test_add_longitude(cubes_2d, tmp_path):
    """Test fix."""
    # Remove longitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('longitude')
    tas_cube.attributes['grid_file_uri'] = (
        'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/'
        'integration/cmor/_fixes/test_data/icon_grid.nc'
    )
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert 'icon_grid.nc' in fix._horizontal_grids

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


def test_add_latitude_longitude(cubes_2d, tmp_path):
    """Test fix."""
    # Remove latitude and longitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.remove_coord('longitude')
    tas_cube.attributes['grid_file_uri'] = (
        'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/'
        'integration/cmor/_fixes/test_data/icon_grid.nc'
    )
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    assert len(fix._horizontal_grids) == 0
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)
    assert cube.shape == (1, 8)
    check_lat_lon(cube)
    assert len(fix._horizontal_grids) == 1
    assert 'icon_grid.nc' in fix._horizontal_grids

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


def test_add_latitude_fail(cubes_2d):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    msg = "Failed to add missing latitude coordinate to cube"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes)


def test_add_longitude_fail(cubes_2d):
    """Test fix."""
    # Remove longitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('longitude')
    cubes = CubeList([tas_cube])
    fix = get_allvars_fix('Amon', 'tas')

    msg = "Failed to add missing longitude coordinate to cube"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes)


def test_add_coord_from_grid_file_fail_invalid_coord():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    msg = r"coord_name must be one of .* got 'invalid_coord_name'"
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(mock.sentinel.cube, 'invalid_coord_name',
                                      'invalid_target_name')


def test_add_coord_from_grid_file_fail_no_url():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    msg = ("Cube does not contain the attribute 'grid_file_uri' necessary to "
           "download the ICON horizontal grid file")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(Cube(0), 'grid_latitude', 'latitude')


def test_add_coord_from_grid_fail_no_unnamed_dim(cubes_2d, tmp_path):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.attributes['grid_file_uri'] = (
        'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/'
        'integration/cmor/_fixes/test_data/icon_grid.nc'
    )
    index_coord = DimCoord(np.arange(8), var_name='ncells')
    tas_cube.add_dim_coord(index_coord, 1)
    fix = get_allvars_fix('Amon', 'tas')

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    msg = ("Cannot determine coordinate dimension for coordinate 'latitude', "
           "cube does not contain a single unnamed dimension")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(tas_cube, 'grid_latitude', 'latitude')

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


def test_add_coord_from_grid_fail_two_unnamed_dims(cubes_2d, tmp_path):
    """Test fix."""
    # Remove latitude from tas cube to test automatic addition
    tas_cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
    tas_cube.remove_coord('latitude')
    tas_cube.attributes['grid_file_uri'] = (
        'https://github.com/ESMValGroup/ESMValCore/raw/main/tests/'
        'integration/cmor/_fixes/test_data/icon_grid.nc'
    )
    tas_cube = iris.util.new_axis(tas_cube)
    fix = get_allvars_fix('Amon', 'tas')

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    msg = ("Cannot determine coordinate dimension for coordinate 'latitude', "
           "cube does not contain a single unnamed dimension")
    with pytest.raises(ValueError, match=msg):
        fix._add_coord_from_grid_file(tas_cube, 'grid_latitude', 'latitude')

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.requests', autospec=True)
def test_get_horizontal_grid_cached_in_dict(mock_requests):
    """Test fix."""
    cube = Cube(0, attributes={'grid_file_uri': 'cached_grid_url.nc'})
    fix = get_allvars_fix('Amon', 'tas')
    fix._horizontal_grids['cached_grid_url.nc'] = mock.sentinel.grid

    grid = fix.get_horizontal_grid(cube)
    assert grid == mock.sentinel.grid
    assert mock_requests.mock_calls == []


@mock.patch('esmvalcore.cmor._fixes.icon._base_fixes.requests', autospec=True)
def test_get_horizontal_grid_cached_in_file(mock_requests, tmp_path):
    """Test fix."""
    cube = Cube(0, attributes={
        'grid_file_uri': 'https://temporary.url/this/is/the/grid_file.nc'})
    fix = get_allvars_fix('Amon', 'tas')
    assert len(fix._horizontal_grids) == 0

    # Save temporary grid file
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, str(tmp_path / 'grid_file.nc'))

    # Temporary overwrite default cache location for downloads
    original_cache_dir = fix.CACHE_DIR
    fix.CACHE_DIR = tmp_path

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 1
    assert grid[0].var_name == 'grid'
    assert len(fix._horizontal_grids) == 1
    assert 'grid_file.nc' in fix._horizontal_grids
    assert mock_requests.mock_calls == []

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir


def test_get_horizontal_grid_cache_file_too_old(tmp_path):
    """Test fix."""
    cube = Cube(0, attributes={
        'grid_file_uri': 'https://github.com/ESMValGroup/ESMValCore/raw/main/'
                         'tests/integration/cmor/_fixes/test_data/'
                         'icon_grid.nc'})
    fix = get_allvars_fix('Amon', 'tas')
    assert len(fix._horizontal_grids) == 0

    # Save temporary grid file
    grid_cube = Cube(0, var_name='grid')
    iris.save(grid_cube, str(tmp_path / 'icon_grid.nc'))

    # Temporary overwrite default cache location for downloads and cache
    # validity duration
    original_cache_dir = fix.CACHE_DIR
    original_cache_validity = fix.CACHE_VALIDITY
    fix.CACHE_DIR = tmp_path
    fix.CACHE_VALIDITY = -1

    grid = fix.get_horizontal_grid(cube)
    assert isinstance(grid, CubeList)
    assert len(grid) == 1
    assert grid[0].var_name == 'cell_area'
    assert len(fix._horizontal_grids) == 1
    assert 'icon_grid.nc' in fix._horizontal_grids

    # Restore cache location
    fix.CACHE_DIR = original_cache_dir
    fix.CACHE_VALIDITY = original_cache_validity


# Test with single-dimension cubes


def test_only_time():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('ICON', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['time']
    extra_facets = get_extra_facets('ICON', 'ICON', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    time_coord = DimCoord([0.0, 1.0], var_name='time', standard_name='time',
                          long_name='time', units='days since 1850-01-01')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(time_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check time metadata
    assert cube.coords('time')
    new_time_coord = cube.coord('time', dim_coords=True)
    assert new_time_coord.var_name == 'time'
    assert new_time_coord.standard_name == 'time'
    assert new_time_coord.long_name == 'time'
    assert new_time_coord.units == 'days since 1850-01-01'

    # Check time data
    np.testing.assert_allclose(new_time_coord.points, [0.0, 1.0])
    np.testing.assert_allclose(new_time_coord.bounds,
                               [[-0.5, 0.5], [0.5, 1.5]])

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


def test_only_height():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('ICON', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['plev19']
    extra_facets = get_extra_facets('ICON', 'ICON', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    height_coord = DimCoord([1000.0, 100.0], var_name='height',
                            standard_name='height', units='cm')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(height_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check height metadata
    assert cube.coords('height', dim_coords=True)
    new_height_coord = cube.coord('height')
    assert new_height_coord.var_name == 'height'
    assert new_height_coord.standard_name == 'height'
    assert new_height_coord.long_name == 'height'
    assert new_height_coord.units == 'm'
    assert new_height_coord.attributes == {'positive': 'up'}

    # Check height data
    np.testing.assert_allclose(new_height_coord.points, [1.0, 10.0])
    assert new_height_coord.bounds is None

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


def test_only_latitude():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('ICON', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['latitude']
    extra_facets = get_extra_facets('ICON', 'ICON', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    lat_coord = DimCoord([0.0, 10.0], var_name='lat', standard_name='latitude',
                         units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(lat_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check latitude metadata
    assert cube.coords('latitude', dim_coords=True)
    new_lat_coord = cube.coord('latitude')
    assert new_lat_coord.var_name == 'lat'
    assert new_lat_coord.standard_name == 'latitude'
    assert new_lat_coord.long_name == 'latitude'
    assert new_lat_coord.units == 'degrees_north'

    # Check latitude data
    np.testing.assert_allclose(new_lat_coord.points, [0.0, 10.0])
    assert new_lat_coord.bounds is None

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


def test_only_longitude():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # ICON CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('ICON', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['longitude']
    extra_facets = get_extra_facets('ICON', 'ICON', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    lon_coord = DimCoord([0.0, 180.0], var_name='lon',
                         standard_name='longitude', units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='ta', units='K',
             dim_coords_and_dims=[(lon_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check longitude metadata
    assert cube.coords('longitude', dim_coords=True)
    new_lon_coord = cube.coord('longitude')
    assert new_lon_coord.var_name == 'lon'
    assert new_lon_coord.standard_name == 'longitude'
    assert new_lon_coord.long_name == 'longitude'
    assert new_lon_coord.units == 'degrees_east'

    # Check longitude data
    np.testing.assert_allclose(new_lon_coord.points, [0.0, 180.0])
    assert new_lon_coord.bounds is None

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


# Test variable not available in file


def test_var_not_available_pr(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'pr')
    msg = "Variable 'pr' used to extract 'pr' is not available in input file"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes_2d)


# Test fix with invalid time units


def test_invalid_time_units(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    for cube in cubes_2d:
        cube.coord('time').attributes['invalid_units'] = 'month as %Y%m%d.%f'
    msg = "Expected time units"
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes_2d)
