"""Tests for the EMAC on-the-fly CMORizer."""
# from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore._config import get_extra_facets
from esmvalcore.cmor._fixes.emac.emac import (
    AllVars,
    Clt,
    Clwvi,
    Evspsbl,
    Hfls,
    Hfss,
    MP_BC_tot,
    MP_DU_tot,
    MP_SO4mm_tot,
    MP_SS_tot,
    Od550aer,
    Pr,
    Rlds,
    Rlus,
    Rlut,
    Rlutcs,
    Rsds,
    Rsdt,
    Rsus,
    Rsut,
    Rsutcs,
    Rtmt,
    Siconc,
    Siconca,
    Sithick,
    Toz,
)
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
def cubes_g3b(test_data_path):
    """g3b sample cubes."""
    nc_path = test_data_path / 'emac_g3b.nc'
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
    assert 'positive' not in cube.attributes
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


def check_plev(cube):
    """Check plev coordinate of cube."""
    assert cube.coords('air_pressure', dim_coords=True)
    plev = cube.coord('air_pressure', dim_coords=True)
    assert plev.var_name == 'plev'
    assert plev.standard_name == 'air_pressure'
    assert plev.long_name == 'pressure'
    assert plev.units == 'Pa'
    assert plev.attributes['positive'] == 'down'

    # Note: plev is reversed (index 0 should be surface, but is TOA at the
    # moment), but this is fixed in the CMOR checks in a later step
    # automatically
    np.testing.assert_allclose(
        plev.points,
        [100.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0,
         15000.0, 20000.0, 25000.0, 30000.0, 40000.0, 50000.0, 60000.0,
         70000.0, 85000.0, 92500.0, 100000.0],
    )
    assert plev.bounds is None


def check_alevel(cube):
    """Check alevel coordinate of cube."""
    # atmosphere_hybrid_sigma_pressure_coordinate
    assert cube.coords('atmosphere_hybrid_sigma_pressure_coordinate',
                       dim_coords=True)
    lev = cube.coord('atmosphere_hybrid_sigma_pressure_coordinate',
                     dim_coords=True)
    assert lev.var_name == 'lev'
    assert lev.standard_name == 'atmosphere_hybrid_sigma_pressure_coordinate'
    assert lev.long_name == 'hybrid sigma pressure coordinate'
    assert lev.units == '1'
    assert lev.attributes['positive'] == 'down'
    np.testing.assert_allclose(
        lev.points[:4],
        [9.96150017e-01, 9.82649982e-01, 9.58960303e-01, 9.27668441e-01],
    )
    np.testing.assert_allclose(
        lev.bounds[:4],
        [[1.00000000e+00, 9.92299974e-01],
         [9.92299974e-01, 9.72999990e-01],
         [9.72999990e-01, 9.44920615e-01],
         [9.44920615e-01, 9.10416267e-01]],
    )

    # Coefficient ap
    assert cube.coords('vertical coordinate formula term: ap(k)',
                       dim_coords=False)
    ap_coord = cube.coord('vertical coordinate formula term: ap(k)',
                          dim_coords=False)
    assert ap_coord.var_name == 'ap'
    assert ap_coord.standard_name is None
    assert ap_coord.long_name == 'vertical coordinate formula term: ap(k)'
    assert ap_coord.units == 'Pa'
    assert ap_coord.attributes == {}
    np.testing.assert_allclose(
        ap_coord.points[:4],
        [0.0, 0.0, 36.03179932, 171.845047],
    )
    np.testing.assert_allclose(
        ap_coord.bounds[:4],
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 72.06359863],
         [72.06359863, 271.62649536]],
    )

    # Coefficient b
    assert cube.coords('vertical coordinate formula term: b(k)',
                       dim_coords=False)
    b_coord = cube.coord('vertical coordinate formula term: b(k)',
                         dim_coords=False)
    assert b_coord.var_name == 'b'
    assert b_coord.standard_name is None
    assert b_coord.long_name == 'vertical coordinate formula term: b(k)'
    assert b_coord.units == '1'
    assert b_coord.attributes == {}
    np.testing.assert_allclose(
        b_coord.points[:4],
        [0.99615002, 0.98264998, 0.95859998, 0.92594999],
    )
    np.testing.assert_allclose(
        b_coord.bounds[:4],
        [[1.0, 0.99229997],
         [0.99229997, 0.97299999],
         [0.97299999, 0.94419998],
         [0.94419998, 0.9077]],
    )

    # Coefficient ps
    assert cube.coords('surface_air_pressure', dim_coords=False)
    ps_coord = cube.coord('surface_air_pressure', dim_coords=False)
    assert ps_coord.var_name == 'ps'
    assert ps_coord.standard_name == 'surface_air_pressure'
    assert ps_coord.long_name == 'Surface Air Pressure'
    assert ps_coord.units == 'Pa'
    assert ps_coord.attributes == {}
    np.testing.assert_allclose(
        ps_coord.points[:, :, 0],
        [[100000.1875, 98240.7578125, 99601.09375, 96029.7109375]],
    )
    assert ps_coord.bounds is None

    # air_pressure
    assert cube.coords('air_pressure', dim_coords=False)
    p_coord = cube.coord('air_pressure', dim_coords=False)
    assert p_coord.var_name is None
    assert p_coord.standard_name == 'air_pressure'
    assert p_coord.long_name is None
    assert p_coord.units == 'Pa'
    assert p_coord.attributes == {}
    assert p_coord.points[0, 0, 0, 0] > p_coord.points[0, -1, 0, 0]
    assert p_coord.bounds[0, 0, 0, 0, 0] > p_coord.bounds[0, -1, 0, 0, 0]
    assert p_coord.bounds[0, 0, 0, 0, 0] > p_coord.bounds[0, 0, 0, 0, 1]


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


def check_lambda550nm(cube):
    """Check scalar lambda550nm coordinate of cube."""
    assert cube.coords('radiation_wavelength')
    typesi = cube.coord('radiation_wavelength')
    assert typesi.var_name == 'wavelength'
    assert typesi.standard_name == 'radiation_wavelength'
    assert typesi.long_name == 'Radiation Wavelength 550 nanometers'
    assert typesi.units == 'nm'
    np.testing.assert_array_equal(typesi.points, [550.0])
    assert typesi.bounds is None


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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[-203.94414, -16.695345, 74.117096, 104.992195]],
        rtol=1e-5,
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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.01435195, 0.006420649, 0.0007885683, 0.01154814]],
        rtol=1e-5,
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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[86.79899, 58.01009, 34.01953, 85.48493]],
        rtol=1e-5,
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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.20945302, 0.01015517, 0.01444221, 0.10618545]],
        rtol=1e-5,
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
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [2.855254e+15, 2.85538e+15],
        rtol=1e-5,
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
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[3.636807e-05, 3.438968e-07, 6.235108e-05, 1.165336e-05]],
        rtol=1e-5,
    )


def test_get_hfls_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hfls')
    assert fix == [Hfls(None), AllVars(None)]


def test_hfls_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'hfls')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'hfls')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'hfls', ())
    fix = Hfls(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'hfls'
    assert cube.standard_name == 'surface_upward_latent_heat_flux'
    assert cube.long_name == 'Surface Upward Latent Heat Flux'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[90.94926, 0.860017, 155.92758, 29.142715]],
        rtol=1e-5,
    )


def test_get_hfss_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hfss')
    assert fix == [Hfss(None), AllVars(None)]


def test_hfss_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'hfss')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'hfss')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'hfss', ())
    fix = Hfss(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'hfss'
    assert cube.standard_name == 'surface_upward_sensible_heat_flux'
    assert cube.long_name == 'Surface Upward Sensible Heat Flux'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[65.92767, 32.841537, 18.461172, 6.50319]],
        rtol=1e-5,
    )


def test_get_od550aer_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'od550aer')
    assert fix == [Od550aer(None), AllVars(None)]


def test_od550aer_fix(cubes_aermon):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'od550aer')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'od550aer', ())
    fix = Od550aer(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_aermon)

    fix = get_allvars_fix('Amon', 'od550aer')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'od550aer'
    assert cube.standard_name == ('atmosphere_optical_thickness_due_to_'
                                  'ambient_aerosol_particles')
    assert cube.long_name == 'Ambient Aerosol Optical Thickness at 550nm'
    assert cube.units == '1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_lambda550nm(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.166031, 0.271185, 0.116384, 0.044266]],
        rtol=1e-5,
    )


def test_get_pr_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'pr')
    assert fix == [Pr(None), AllVars(None)]


def test_pr_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'pr')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'pr', ())
    fix = Pr(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'pr')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'pr'
    assert cube.standard_name == 'precipitation_flux'
    assert cube.long_name == 'Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[3.590828e-05, 5.637868e-07, 3.474401e-07, 1.853631e-05]],
        rtol=1e-5,
    )


def test_get_prc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prc')
    assert fix == [AllVars(None)]


def test_prc_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'prc')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prc'
    assert cube.standard_name == 'convective_precipitation_flux'
    assert cube.long_name == 'Convective Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[1.177248e-05, 0.0, 0.0, 2.419762e-06]],
        rtol=1e-5,
    )


def test_get_prl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prl')
    assert fix == [AllVars(None)]


def test_prl_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'prl')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prl'
    assert cube.standard_name is None
    assert cube.long_name == 'Large Scale Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[2.091789e-05, 5.637868e-07, 3.474401e-07, 1.611654e-05]],
        rtol=1e-5,
    )


def test_get_prsn_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prsn')
    assert fix == [AllVars(None)]


def test_prsn_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'prsn')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prsn'
    assert cube.standard_name == 'snowfall_flux'
    assert cube.long_name == 'Snowfall Flux'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[3.217916e-06, 5.760116e-30, 5.894975e-30, 6.625394e-30]],
        rtol=1e-5,
    )


def test_get_prw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prw')
    assert fix == [AllVars(None)]


def test_prw_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'prw')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prw'
    assert cube.standard_name == 'atmosphere_mass_content_of_water_vapor'
    assert cube.long_name == 'Water Vapor Path'
    assert cube.units == 'kg m-2'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[9.398615, 10.207355, 22.597773, 11.342406]],
        rtol=1e-5,
    )


def test_get_rlds_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlds')
    assert fix == [Rlds(None), AllVars(None)]


def test_rlds_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'rlds')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rlds', ())
    fix = Rlds(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'rlds')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rlds'
    assert cube.standard_name == 'surface_downwelling_longwave_flux_in_air'
    assert cube.long_name == 'Surface Downwelling Longwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[297.55298, 310.508, 361.471, 302.51376]],
        rtol=1e-5,
    )


def test_get_rlus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlus')
    assert fix == [Rlus(None), AllVars(None)]


def test_rlus_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rlus')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rlus')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rlus', ())
    fix = Rlus(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rlus'
    assert cube.standard_name == 'surface_upwelling_longwave_flux_in_air'
    assert cube.long_name == 'Surface Upwelling Longwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[351.59143, 411.6364, 438.25314, 339.71625]],
        rtol=1e-5,
    )


def test_get_rlut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlut')
    assert fix == [Rlut(None), AllVars(None)]


def test_rlut_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rlut')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rlut')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rlut', ())
    fix = Rlut(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rlut'
    assert cube.standard_name == 'toa_outgoing_longwave_flux'
    assert cube.long_name == 'TOA Outgoing Longwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[181.34714, 240.24974, 282.01166, 203.07207]],
        rtol=1e-5,
    )


def test_get_rlutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlutcs')
    assert fix == [Rlutcs(None), AllVars(None)]


def test_rlutcs_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rlutcs')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rlutcs')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rlutcs', ())
    fix = Rlutcs(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rlutcs'
    assert cube.standard_name == ('toa_outgoing_longwave_flux_assuming_clear_'
                                  'sky')
    assert cube.long_name == 'TOA Outgoing Clear-Sky Longwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[232.35957, 273.42227, 288.2262, 238.6909]],
        rtol=1e-5,
    )


def test_get_rsds_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsds')
    assert fix == [Rsds(None), AllVars(None)]


def test_rsds_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'rsds')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsds', ())
    fix = Rsds(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'rsds')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsds'
    assert cube.standard_name == 'surface_downwelling_shortwave_flux_in_air'
    assert cube.long_name == 'Surface Downwelling Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[7.495961, 214.6077, 349.77203, 191.22644]],
        rtol=1e-5,
    )


def test_get_rsdt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsdt')
    assert fix == [Rsdt(None), AllVars(None)]


def test_rsdt_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'rsdt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsdt', ())
    fix = Rsdt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'rsdt')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsdt'
    assert cube.standard_name == 'toa_incoming_shortwave_flux'
    assert cube.long_name == 'TOA Incident Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[44.4018, 312.62286, 481.91992, 473.25092]],
        rtol=1e-5,
    )


def test_get_rsus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsus')
    assert fix == [Rsus(None), AllVars(None)]


def test_rsus_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsus')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rsus')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsus', ())
    fix = Rsus(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rsus'
    assert cube.standard_name == 'surface_upwelling_shortwave_flux_in_air'
    assert cube.long_name == 'Surface Upwelling Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[0.524717, 82.92702, 24.484043, 13.38585]],
        rtol=1e-5,
    )


def test_get_rsut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsut')
    assert fix == [Rsut(None), AllVars(None)]


def test_rsut_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsut')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rsut')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsut', ())
    fix = Rsut(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rsut'
    assert cube.standard_name == 'toa_outgoing_shortwave_flux'
    assert cube.long_name == 'TOA Outgoing Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[26.967886, 114.11882, 70.44302, 203.26039]],
        rtol=1e-5,
    )


def test_get_rsutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsutcs')
    assert fix == [Rsutcs(None), AllVars(None)]


def test_rsutcs_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'rsutcs')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'Amon', 'rsutcs')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsutcs', ())
    fix = Rsutcs(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'rsutcs'
    assert cube.standard_name == ('toa_outgoing_shortwave_flux_assuming_clear_'
                                  'sky')
    assert cube.long_name == 'TOA Outgoing Clear-Sky Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'up'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[11.787124, 101.68645, 50.588364, 53.933403]],
        rtol=1e-5,
    )


def test_get_rtmt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rtmt')
    assert fix == [Rtmt(None), AllVars(None)]


def test_rtmt_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'Amon', 'rtmt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rtmt', ())
    fix = Rtmt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('Amon', 'rtmt')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rtmt'
    assert cube.standard_name == ('net_downward_radiative_flux_at_top_of_'
                                  'atmosphere_model')
    assert cube.long_name == 'Net Downward Radiative Flux at Top of Model'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[-163.91322, -41.745697, 129.46524, 66.91847]],
        rtol=1e-5,
    )


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconc')
    assert fix == [Siconc(None), AllVars(None)]


def test_siconc_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'SImon', 'siconc')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconc', ())
    fix = Siconc(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('SImon', 'siconc')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'siconc'
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == 'Sea-Ice Area Percentage (Ocean Grid)'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 1],
        [[61.51324, 0.0, 0.0, 0.0]],
        rtol=1e-5,
    )


def test_get_siconca_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconca')
    assert fix == [Siconca(None), AllVars(None)]


def test_siconca_fix(cubes_amon_2d):
    """Test fix."""
    vardef = get_var_info('EMAC', 'SImon', 'siconca')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconca', ())
    fix = Siconca(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    fix = get_allvars_fix('SImon', 'siconca')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'siconca'
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == 'Sea-Ice Area Percentage (Atmospheric Grid)'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 1],
        [[61.51324, 0.0, 0.0, 0.0]],
        rtol=1e-5,
    )


def test_get_sithick_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'sithick')
    assert fix == [Sithick(None), AllVars(None)]


def test_sithick_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('SImon', 'sithick')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]

    vardef = get_var_info('EMAC', 'SImon', 'sithick')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'sithick', ())
    fix = Sithick(vardef, extra_facets=extra_facets)
    cube = fix.fix_data(cube)

    assert cube.var_name == 'sithick'
    assert cube.standard_name == 'sea_ice_thickness'
    assert cube.long_name == 'Sea Ice Thickness'
    assert cube.units == 'm'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(cube.data[0, 0, 1], 0.798652, rtol=1e-5,)
    np.testing.assert_equal(
        cube.data[:, :, 1].mask,
        [[False, True, True, True]],
    )


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_tas_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_heightxm(cube, 2.0)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[277.4016, 291.2251, 295.6336, 277.8235]],
        rtol=1e-5,
    )


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Omon', 'tos')
    assert fix == [AllVars(None)]


def test_tos_fix(cubes_g3b):
    """Test fix."""
    fix = get_allvars_fix('Omon', 'tos')
    fixed_cubes = fix.fix_metadata(cubes_g3b)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tos'
    assert cube.standard_name == 'sea_surface_temperature'
    assert cube.long_name == 'Sea Surface Temperature'
    assert cube.units == 'degC'
    assert 'positive' not in cube.attributes

    print(cube.coord('time'))

    check_time(cube, n_points=2)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[0, :, 0],
        [7.828393, 10.133539, 23.036158, 4.997858],
        rtol=1e-5,
    )


def test_get_toz_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'AERmon', 'toz')
    assert fix == [Toz(None), AllVars(None)]


def test_toz_fix(cubes_column):
    """Test fix."""
    vardef = get_var_info('EMAC', 'AERmon', 'toz')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'AERmon', 'toz', ())
    fix = Toz(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_column)

    fix = get_allvars_fix('AERmon', 'toz')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'toz'
    assert cube.standard_name == ('equivalent_thickness_at_stp_of_atmosphere_'
                                  'ozone_content')
    assert cube.long_name == 'Total Column Ozone'
    assert cube.units == 'm'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[0, :, 0],
        [0.003108, 0.002928, 0.002921, 0.003366],
        rtol=1e-3,
    )


def test_get_ts_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ts')
    assert fix == [AllVars(None)]


def test_ts_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ts')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ts'
    assert cube.standard_name == 'surface_temperature'
    assert cube.long_name == 'Surface Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[280.65475, 291.80563, 296.55356, 278.24164]],
        rtol=1e-5,
    )


def test_get_uas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'uas')
    assert fix == [AllVars(None)]


def test_uas_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'uas')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'uas'
    assert cube.standard_name == 'eastward_wind'
    assert cube.long_name == 'Eastward Near-Surface Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_heightxm(cube, 10.0)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[-2.114626, -2.809653, -6.59721, -1.586884]],
        rtol=1e-5,
    )


def test_get_vas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'vas')
    assert fix == [AllVars(None)]


def test_vas_fix(cubes_amon_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'vas')
    fixed_cubes = fix.fix_metadata(cubes_amon_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'vas'
    assert cube.standard_name == 'northward_wind'
    assert cube.long_name == 'Northward Near-Surface Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_time(cube)
    check_lat(cube)
    check_lon(cube)
    check_heightxm(cube, 10.0)

    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[3.026835, -2.226409, 4.868941, 3.301589]],
        rtol=1e-5,
    )


# Test each tracer variable in extra_facets/emac-mappings.yml


def test_get_MP_BC_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_BC_tot')
    assert fix == [MP_BC_tot(None), AllVars(None)]


def test_MP_BC_tot_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_BC_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_BC_tot',
                                    ())
    fix = MP_BC_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    fix = get_allvars_fix('TRAC10hr', 'MP_BC_tot')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_BC_tot'
    assert cube.standard_name is None
    assert cube.long_name == ('total mass of black carbon (sum of all aerosol '
                              'modes)')
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [6.361834e+08, 6.371043e+08],
        rtol=1e-5,
    )


def test_get_MP_CFCl3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CFCl3')
    assert fix == [AllVars(None)]


def test_MP_CFCl3_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_CFCl3')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CFCl3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CFCl3 (CFC-11)'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [5.982788e+09, 5.982657e+09],
        rtol=1e-5,
    )


def test_get_MP_ClOX_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_ClOX')
    assert fix == [AllVars(None)]


def test_MP_ClOX_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_ClOX')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_ClOX'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of ClOX'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [39589028.0, 39722044.0],
        rtol=1e-5,
    )


def test_get_MP_CH4_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CH4')
    assert fix == [AllVars(None)]


def test_MP_CH4_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_CH4')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CH4'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CH4'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [4.866472e+12, 4.866396e+12],
        rtol=1e-5,
    )


def test_get_MP_CO_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CO')
    assert fix == [AllVars(None)]


def test_MP_CO_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_CO')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CO'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CO'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [3.399702e+11, 3.401483e+11],
        rtol=1e-5,
    )


def test_get_MP_CO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CO2')
    assert fix == [AllVars(None)]


def test_MP_CO2_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_CO2')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [2.855254e+15, 2.855380e+15],
        rtol=1e-5,
    )


def test_get_MP_DU_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_DU_tot')
    assert fix == [MP_DU_tot(None), AllVars(None)]


def test_MP_DU_tot_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_DU_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_DU_tot',
                                    ())
    fix = MP_DU_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    fix = get_allvars_fix('TRAC10hr', 'MP_DU_tot')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_DU_tot'
    assert cube.standard_name is None
    assert cube.long_name == ('total mass of mineral dust (sum of all aerosol '
                              'modes)')
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [1.797283e+10, 1.704390e+10],
        rtol=1e-5,
    )


def test_get_MP_N2O_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_N2O')
    assert fix == [AllVars(None)]


def test_MP_N2O_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_N2O')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_N2O'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of N2O'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [2.365061e+12, 2.365089e+12],
        rtol=1e-5,
    )


def test_get_MP_NH3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NH3')
    assert fix == [AllVars(None)]


def test_MP_NH3_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_NH3')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NH3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NH3'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [1.931037e+08, 1.944860e+08],
        rtol=1e-5,
    )


def test_get_MP_NO_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NO')
    assert fix == [AllVars(None)]


def test_MP_NO_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_NO')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NO'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NO'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [5.399146e+08, 5.543320e+08],
        rtol=1e-5,
    )


def test_get_MP_NO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NO2')
    assert fix == [AllVars(None)]


def test_MP_NO2_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_NO2')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [1.734202e+09, 1.725541e+09],
        rtol=1e-5,
    )


def test_get_MP_NOX_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NOX')
    assert fix == [AllVars(None)]


def test_MP_NOX_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_NOX')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NOX'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NOX (NO+NO2)'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [9.384478e+08, 9.342440e+08],
        rtol=1e-5,
    )


def test_get_MP_O3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_O3')
    assert fix == [AllVars(None)]


def test_MP_O3_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_O3')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_O3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of O3'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [3.339367e+12, 3.339434e+12],
        rtol=1e-5,
    )


def test_get_MP_OH_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_OH')
    assert fix == [AllVars(None)]


def test_MP_OH_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_OH')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_OH'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of OH'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [3816360.8, 3820260.8],
        rtol=1e-5,
    )


def test_get_MP_S_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_S')
    assert fix == [AllVars(None)]


def test_MP_S_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_S')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_S'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of S'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [0.0, 0.0],
        rtol=1e-5,
    )


def test_get_MP_SO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO2')
    assert fix == [AllVars(None)]


def test_MP_SO2_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    fix = get_allvars_fix('TRAC10hr', 'MP_SO2')
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_SO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of SO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [1.383063e+09, 1.390189e+09],
        rtol=1e-5,
    )


def test_get_MP_SO4mm_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO4mm_tot')
    assert fix == [MP_SO4mm_tot(None), AllVars(None)]


def test_MP_SO4mm_tot_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_SO4mm_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO4mm_tot',
                                    ())
    fix = MP_SO4mm_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    fix = get_allvars_fix('TRAC10hr', 'MP_SO4mm_tot')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_SO4mm_tot'
    assert cube.standard_name is None
    assert cube.long_name == ('total mass of aerosol sulfate (sum of all '
                              'aerosol modes)')
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [1.350434e+09, 1.364699e+09],
        rtol=1e-5,
    )


def test_get_MP_SS_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SS_tot')
    assert fix == [MP_SS_tot(None), AllVars(None)]


def test_MP_SS_tot_fix(cubes_tracer_pdef_gp):  # noqa: N802
    """Test fix."""
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_SS_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_SS_tot',
                                    ())
    fix = MP_SS_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_tracer_pdef_gp)

    fix = get_allvars_fix('TRAC10hr', 'MP_SS_tot')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_SS_tot'
    assert cube.standard_name is None
    assert cube.long_name == ('total mass of sea salt (sum of all aerosol '
                              'modes)')
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    check_time(cube, n_points=2)

    np.testing.assert_allclose(
        cube.data,
        [2.322862e+08, 2.340771e+08],
        rtol=1e-5,
    )


# Test each 3D variable with regular Z-coord in extra_facets/emac-mappings.yml


def test_get_ta_amon_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ta')
    assert fix == [AllVars(None)]


def test_ta_amon_fix(cubes_amon_3d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    fixed_cubes = fix.fix_metadata(cubes_amon_3d)

    cube = check_ta_metadata(fixed_cubes)

    fixed_cube = fix.fix_data(cube)

    check_time(fixed_cube)
    check_plev(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)

    np.testing.assert_allclose(
        fixed_cube.data[0, -5:-1, 0, 0],
        [250.93347, 258.48843, 266.4087, 270.26993],
        rtol=1e-5,
    )
    np.testing.assert_equal(
        fixed_cube.data.mask[0, -5:, 0, 0],
        [False, False, False, False, True],
    )


# Test each 3D variable with hybrid Z-coord in extra_facets/emac-mappings.yml


def test_get_ta_cfon_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'CFmon', 'ta')
    assert fix == [AllVars(None)]


def test_ta_cfmon_fix(test_data_path, tmp_path):
    """Test fix."""
    fix = get_allvars_fix('CFmon', 'ta')

    filepath = test_data_path / 'emac_amon_3d.nc'
    fixed_path = fix.fix_file(filepath, tmp_path)
    cubes = iris.load(fixed_path)

    assert cubes.extract(NameConstraint(var_name='hyam'))
    assert cubes.extract(NameConstraint(var_name='hybm'))
    assert cubes.extract(NameConstraint(var_name='hyai'))
    assert cubes.extract(NameConstraint(var_name='hybi'))

    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_ta_metadata(fixed_cubes)

    fixed_cube = fix.fix_data(cube)

    check_time(fixed_cube)
    check_alevel(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)

    np.testing.assert_allclose(
        fixed_cube.data[0, 0:5, 0, 0],
        [272.32098, 271.45898, 270.3698, 269.20953, 267.84683],
        rtol=1e-5,
    )


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
