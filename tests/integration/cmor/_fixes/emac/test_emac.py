"""Tests for the EMAC on-the-fly CMORizer."""
# from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
# from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore._config import get_extra_facets
from esmvalcore.cmor._fixes.emac.emac import AllVars, Clt, Clwvi, Evspsbl
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info

# Note: test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


@pytest.fixture
def cubes_aermon(test_data_path):
    """AERmon sample cubes."""
    nc_path = test_data_path / 'emac_aermon.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_amon_2d(test_data_path):
    """Amon 2D sample cubes."""
    nc_path = test_data_path / 'emac_amon_2d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_amon_3d(test_data_path):
    """Amon 3D sample cubes."""
    nc_path = test_data_path / 'emac_amon_3d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_column(test_data_path):
    """column sample cubes."""
    nc_path = test_data_path / 'emac_column.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_omon_2d(test_data_path):
    """Omon 2D sample cubes."""
    nc_path = test_data_path / 'emac_omon_2d.nc'
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_tracer_pdef_gp(test_data_path):
    """tracer_pdef_gp sample cubes."""
    nc_path = test_data_path / 'emac_tracer_pdef_gp.nc'
    return iris.load(str(nc_path))


def get_allvars_fix(mip, short_name):
    """Get member of fix class."""
    vardef = get_var_info('EMAC', mip, short_name)
    extra_facets = get_extra_facets('EMAC', 'EMAC', mip, short_name, ())
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
    return cube


def check_tas_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    return cube


def check_siconc_metadata(cubes, var_name, long_name):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == var_name
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == long_name
    assert cube.units == '%'
    return cube


def check_time(cube, n_points=1):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('day since 1849-01-01 00:00:00',
                              calendar='gregorian')
    if n_points == 1:
        np.testing.assert_allclose(time.points, [55181.9930555556])
        assert time.bounds is None
    elif n_points == 2:
        np.testing.assert_allclose(time.points, [55151.25, 55151.666667])
        np.testing.assert_allclose(
            time.bounds,
            [[55151.04166667, 55151.45833333], [55151.45833333, 55151.875]],
        )
    else:
        assert False, "Invalid n_points"
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

    if plev_has_bounds:
        assert plev.bounds is not None
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
    assert cube.coords('latitude', dim_coords=True)
    lat = cube.coord('latitude', dim_coords=True)
    assert lat.var_name == 'lat'
    assert lat.standard_name == 'latitude'
    assert lat.long_name == 'latitude'
    assert lat.units == 'degrees_north'
    np.testing.assert_allclose(
        lat.points,
        [59.4444082891668, 19.8757191474409, -19.8757191474409,
         -59.4444082891668],
    )
    np.testing.assert_allclose(
        lat.bounds,
        [[79.22875286, 39.66006372],
         [39.66006372, 0.0],
         [0.0, -39.66006372],
         [-39.66006372, -79.22875286]],
    )
    assert lat.attributes == {}


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords('longitude', dim_coords=True)
    lon = cube.coord('longitude', dim_coords=True)
    assert lon.var_name == 'lon'
    assert lon.standard_name == 'longitude'
    assert lon.long_name == 'longitude'
    assert lon.units == 'degrees_east'
    np.testing.assert_allclose(
        lon.points,
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    )
    np.testing.assert_allclose(
        lon.bounds,
        [[-22.5, 22.5], [22.5, 67.5], [67.5, 112.5], [112.5, 157.5],
         [157.5, 202.5], [202.5, 247.5], [247.5, 292.5], [292.5, 337.5]],
    )
    assert lon.attributes == {}


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


# Test with single-dimension cubes


def test_only_time():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # EMAC CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('EMAC', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['time']
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    time_coord = DimCoord([0.0, 1.0], var_name='time', standard_name='time',
                          long_name='time', units='days since 1850-01-01')
    cubes = CubeList([
        Cube([1, 1], var_name='tm1_p19_ave', units='K',
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


def test_only_plev():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # EMAC CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('EMAC', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['plev19']
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    plev_coord = DimCoord([1000.0, 900.0], var_name='plev',
                          standard_name='air_pressure', units='hPa')
    cubes = CubeList([
        Cube([1, 1], var_name='tm1_p19_ave', units='K',
             dim_coords_and_dims=[(plev_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_ta_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (2,)
    np.testing.assert_equal(cube.data, [1, 1])

    # Check plev metadata
    assert cube.coords('air_pressure', dim_coords=True)
    new_plev_coord = cube.coord('air_pressure')
    assert new_plev_coord.var_name == 'plev'
    assert new_plev_coord.standard_name == 'air_pressure'
    assert new_plev_coord.long_name == 'pressure'
    assert new_plev_coord.units == 'Pa'
    assert new_plev_coord.attributes == {'positive': 'down'}

    # Check plev data
    np.testing.assert_allclose(new_plev_coord.points, [100000.0, 90000.0])
    assert new_plev_coord.bounds is None

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


def test_only_latitude():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # EMAC CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('EMAC', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['latitude']
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    lat_coord = DimCoord([0.0, 10.0], var_name='lat', standard_name='latitude',
                         units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='tm1_p19_ave', units='K',
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
    np.testing.assert_allclose(new_lat_coord.bounds,
                               [[-5.0, 5.0], [5.0, 15.0]])

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


def test_only_longitude():
    """Test fix."""
    # We know that ta has dimensions time, plev19, latitude, longitude, but the
    # EMAC CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of ta to create
    # an artificial, but realistic test case.
    vardef = get_var_info('EMAC', 'Amon', 'ta')
    original_dimensions = vardef.dimensions
    vardef.dimensions = ['longitude']
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'ta', ())
    fix = AllVars(vardef, extra_facets=extra_facets)

    # Create cube with only a single dimension
    lon_coord = DimCoord([0.0, 180.0], var_name='lon',
                         standard_name='longitude', units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='tm1_p19_ave', units='K',
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
    np.testing.assert_allclose(new_lon_coord.bounds,
                               [[-90.0, 90.0], [90.0, 270.0]])

    # Restore original dimensions of ta
    vardef.dimensions = original_dimensions


# Test each 2D variable in extra_facets/emac-mappings.yml


def test_get_awhea_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Omon', 'awhea')
    assert fix == [AllVars(None)]


def test_awhea_fix(cubes_omon_2d):
    """Test fix."""
    fix = get_allvars_fix('Omon', 'awhea')
    fixed_cubes = fix.fix_metadata(cubes_omon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'awhea'
    assert cube.standard_name is None
    assert cube.long_name == ('Global Mean Net Surface Heat Flux Over Open '
                              'Water')
    assert cube.units == 'W m-2'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[-203.94414, -16.695345, 74.117096, 104.992195]],
        rtol=1e-6,
    )


def test_get_clivi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clivi')
    assert fix == [AllVars(None)]


def test_clivi_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'clivi')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clivi'
    assert cube.standard_name == 'atmosphere_mass_content_of_cloud_ice'
    assert cube.long_name == 'Ice Water Path'
    assert cube.units == 'kg m-2'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.01435195, 0.006420649, 0.0007885683, 0.01154814]],
        rtol=1e-6,
    )


def test_get_clt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clt')
    assert fix == [Clt(None), AllVars(None)]


def test_clt_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'clt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'clt', ())
    fix = Clt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'clt')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clt'
    assert cube.standard_name == 'cloud_area_fraction'
    assert cube.long_name == 'Total Cloud Cover Percentage'
    assert cube.units == '%'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[86.79899, 58.01009, 34.01953, 85.48493]],
        rtol=1e-6,
    )


def test_get_clwvi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clwvi')
    assert fix == [Clwvi(None), AllVars(None)]


def test_clwvi_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'clwvi')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'clwvi', ())
    fix = Clwvi(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'clwvi')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clwvi'
    assert cube.standard_name == ('atmosphere_mass_content_of_cloud_'
                                  'condensed_water')
    assert cube.long_name == 'Condensed Water Path'
    assert cube.units == 'kg m-2'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.20945302, 0.01015517, 0.01444221, 0.10618545]],
        rtol=1e-6,
    )


def test_get_co2mass_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'co2mass')
    assert fix == [AllVars(None)]


def test_co2mass_fix(cubes_tracer_pdef_gp):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'co2mass')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'co2mass'
    assert cube.standard_name == 'atmosphere_mass_of_carbon_dioxide'
    assert cube.long_name == 'Total Atmospheric Mass of CO2'
    assert cube.units == 'kg'

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [2.855254e+15, 2.85538e+15],
        rtol=1e-6,
    )


def test_get_evspsbl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def test_evspsbl_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'evspsbl')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'evspsbl')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'evspsbl', ())
    fix = Evspsbl(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'evspsbl'
    assert cube.standard_name == 'water_evapotranspiration_flux'
    assert cube.long_name == ('Evaporation Including Sublimation and '
                              'Transpiration')
    assert cube.units == 'kg m-2 s-1'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[3.636807e-05, 3.438968e-07, 6.235108e-05, 1.165336e-05]],
        rtol=1e-6,
    )


# Test areacella and areacello (for extra_facets, and grid_latitude and
# grid_longitude coordinates)


# def test_get_areacella_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'fx', 'areacella')
#     assert fix == [AllVars(None)]


# def test_areacella_fix(cubes_grid):
#     """Test fix."""
#     fix = get_allvars_fix('fx', 'areacella')
#     fixed_cubes = fix.fix_metadata(cubes_grid)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.var_name == 'areacella'
#     assert cube.standard_name == 'cell_area'
#     assert cube.long_name == 'Grid-Cell Area for Atmospheric Grid Variables'
#     assert cube.units == 'm2'

#     check_lat_lon(cube)


# def test_get_areacello_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'Ofx', 'areacello')
#     assert fix == [AllVars(None)]


# def test_areacello_fix(cubes_grid):
#     """Test fix."""
#     fix = get_allvars_fix('Ofx', 'areacello')
#     fixed_cubes = fix.fix_metadata(cubes_grid)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.var_name == 'areacello'
#     assert cube.standard_name == 'cell_area'
#     assert cube.long_name == 'Grid-Cell Area for Ocean Variables'
#     assert cube.units == 'm2'

#     check_lat_lon(cube)


# # Test clwvi (for extra_facets)


# def test_get_clwvi_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clwvi')
#     assert fix == [AllVars(None)]


# def test_clwvi_fix(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'clwvi')
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.var_name == 'clwvi'
#     assert cube.standard_name == ('atmosphere_mass_content_of_cloud_'
#                                   'condensed_water')
#     assert cube.long_name == 'Condensed Water Path'
#     assert cube.units == 'kg m-2'

#     check_time(cube)
#     check_lat_lon(cube)


# # Test siconc and siconca (for extra_facets, extra fix and typesi coordinate)


# def test_get_siconc_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconc')
#     assert fix == [Siconc(None), AllVars(None)]


# def test_siconc_fix(cubes_2d):
#     """Test fix."""
#     vardef = get_var_info('EMAC', 'SImon', 'siconc')
#     extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconc', ())
#     siconc_fix = Siconc(vardef, extra_facets=extra_facets)
#     allvars_fix = get_allvars_fix('SImon', 'siconc')

#     fixed_cubes = siconc_fix.fix_metadata(cubes_2d)
#     fixed_cubes = allvars_fix.fix_metadata(fixed_cubes)

#     cube = check_siconc_metadata(fixed_cubes, 'siconc',
#                                  'Sea-Ice Area Percentage (Ocean Grid)')
#     check_time(cube)
#     check_lat_lon(cube)
#     check_typesi(cube)

#     np.testing.assert_allclose(
#         cube.data,
#         [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
#     )


# def test_get_siconca_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconca')
#     assert fix == [Siconca(None), AllVars(None)]


# def test_siconca_fix(cubes_2d):
#     """Test fix."""
#     vardef = get_var_info('EMAC', 'SImon', 'siconca')
#     extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconca', ())
#     siconca_fix = Siconca(vardef, extra_facets=extra_facets)
#     allvars_fix = get_allvars_fix('SImon', 'siconca')

#     fixed_cubes = siconca_fix.fix_metadata(cubes_2d)
#     fixed_cubes = allvars_fix.fix_metadata(fixed_cubes)

#     cube = check_siconc_metadata(
#       fixed_cubes, 'siconca', 'Sea-Ice Area Percentage (Atmospheric Grid)')
#     check_time(cube)
#     check_lat_lon(cube)
#     check_typesi(cube)

#     np.testing.assert_allclose(
#         cube.data,
#         [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
#     )


# # Test ta (for height and plev coordinate)


# def test_get_ta_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ta')
#     assert fix == [AllVars(None)]


# def test_ta_fix(cubes_3d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'ta')
#     fixed_cubes = fix.fix_metadata(cubes_3d)

#     cube = check_ta_metadata(fixed_cubes)
#     check_time(cube)
#     check_height(cube)
#     check_lat_lon(cube)


# def test_ta_fix_no_plev_bounds(cubes_3d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'ta')
#     cubes = CubeList([
#         cubes_3d.extract_cube(NameConstraint(var_name='ta')),
#         cubes_3d.extract_cube(NameConstraint(var_name='pfull')),
#     ])
#     fixed_cubes = fix.fix_metadata(cubes)

#     cube = check_ta_metadata(fixed_cubes)
#     check_time(cube)
#     check_height(cube, plev_has_bounds=False)
#     check_lat_lon(cube)


# # Test tas (for height2m coordinate)


# def test_get_tas_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tas')
#     assert fix == [AllVars(None)]


# def test_tas_fix(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'tas')
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     cube = check_tas_metadata(fixed_cubes)
#     check_time(cube)
#     check_lat_lon(cube)
#     check_heightxm(cube, 2.0)


# def test_tas_spatial_index_coord_already_present(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'tas')

#     index_coord = DimCoord(np.arange(8), var_name='ncells')
#     cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
#     cube.add_dim_coord(index_coord, 1)
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     check_lat_lon(cube)


# def test_tas_scalar_height2m_already_present(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'tas')

#     # Scalar height (with wrong metadata) already present
#     height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
#     cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
#     cube.add_aux_coord(height_coord, ())
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.shape == (1, 8)
#     check_heightxm(cube, 2.0)


# def test_tas_dim_height2m_already_present(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'tas')

#     # Dimensional coordinate height (with wrong metadata) already present
#     height_coord = AuxCoord(2.0, var_name='h', standard_name='height')
#     cube = cubes_2d.extract_cube(NameConstraint(var_name='tas'))
#     cube.add_aux_coord(height_coord, ())
#     cube = iris.util.new_axis(cube, scalar_coord='height')
#     cube.transpose((1, 0, 2))
#     cubes = CubeList([cube])
#     fixed_cubes = fix.fix_metadata(cubes)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.shape == (1, 8)
#     check_heightxm(cube, 2.0)


# # Test uas (for height10m coordinate)


# def test_get_uas_fix():
#     """Test getting of fix."""
#     fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'uas')
#     assert fix == [AllVars(None)]


# def test_uas_fix(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'uas')
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.var_name == 'uas'
#     assert cube.standard_name == 'eastward_wind'
#     assert cube.long_name == 'Eastward Near-Surface Wind'
#     assert cube.units == 'm s-1'

#     check_time(cube)
#     check_lat_lon(cube)
#     assert cube.coords('height')
#     height = cube.coord('height')
#     assert height.var_name == 'height'
#     assert height.standard_name == 'height'
#     assert height.long_name == 'height'
#     assert height.units == 'm'
#     assert height.attributes == {'positive': 'up'}
#     np.testing.assert_allclose(height.points, [10.0])
#     assert height.bounds is None


# def test_uas_scalar_height10m_already_present(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'uas')

#     # Scalar height (with wrong metadata) already present
#     height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
#     cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
#     cube.add_aux_coord(height_coord, ())
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.shape == (1, 8)
#     check_heightxm(cube, 10.0)


# def test_uas_dim_height10m_already_present(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'uas')

#     # Dimensional coordinate height (with wrong metadata) already present
#     height_coord = AuxCoord(10.0, var_name='h', standard_name='height')
#     cube = cubes_2d.extract_cube(NameConstraint(var_name='uas'))
#     cube.add_aux_coord(height_coord, ())
#     cube = iris.util.new_axis(cube, scalar_coord='height')
#     cube.transpose((1, 0, 2))
#     cubes = CubeList([cube])
#     fixed_cubes = fix.fix_metadata(cubes)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.shape == (1, 8)
#     check_heightxm(cube, 10.0)


# # Test fix with empty standard_name


# def test_empty_standard_name_fix(cubes_2d):
#     """Test fix."""
#     # We know that tas has a standard name, but this being native model
#     # output
#     # there may be variables with no standard name. The code is designed to
#     # handle this gracefully and here we test it with an artificial, but
#     # realistic case.
#     vardef = get_var_info('EMAC', 'Amon', 'tas')
#     original_standard_name = vardef.standard_name
#     vardef.standard_name = ''
#     extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'tas', ())
#     fix = AllVars(vardef, extra_facets=extra_facets)
#     fixed_cubes = fix.fix_metadata(cubes_2d)

#     assert len(fixed_cubes) == 1
#     cube = fixed_cubes[0]
#     assert cube.var_name == 'tas'
#     assert cube.standard_name is None
#     assert cube.long_name == 'Near-Surface Air Temperature'
#     assert cube.units == 'K'

#     # Restore original standard_name of tas
#     vardef.standard_name = original_standard_name


# # Test variable not available in file


# def test_var_not_available_pr(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'pr')
#     msg = "Variable 'pr' used to extract 'pr' is not available in input file"
#     with pytest.raises(ValueError, match=msg):
#         fix.fix_metadata(cubes_2d)


# def test_var_not_available_ps(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'ps')
#     msg = "Variable 'x' used to extract 'ps' is not available in input file"
#     with pytest.raises(ValueError, match=msg):
#         fix.get_cube(cubes_2d, var_name='x')


# # Test fix with invalid time units


# def test_invalid_time_units(cubes_2d):
#     """Test fix."""
#     fix = get_allvars_fix('Amon', 'tas')
#     for cube in cubes_2d:
#         cube.coord('time').attributes['invalid_units'] = 'month as %Y%m%d.%f'
#     msg = "Expected time units"
#     with pytest.raises(ValueError, match=msg):
#         fix.fix_metadata(cubes_2d)
