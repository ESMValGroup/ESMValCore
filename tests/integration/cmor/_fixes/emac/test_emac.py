"""Tests for the EMAC on-the-fly CMORizer."""
from unittest import mock

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore._config import get_extra_facets
from esmvalcore.cmor._fixes.emac.emac import (
    AllVars,
    Cl,
    Clt,
    Clwvi,
    Evspsbl,
    Hfls,
    Hfss,
    Hurs,
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
    Zg,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes_1d():
    """1D cube."""
    time_coord = DimCoord(
        0.0,
        var_name='time',
        long_name='time',
        units=Unit('day since 1950-01-01 00:00:00', calendar='gregorian'),
    )
    cube = Cube([1.0], dim_coords_and_dims=[(time_coord, 0)])
    cubes = CubeList([
        cube.copy(),
        cube.copy(),
        cube.copy(),
        cube.copy(),
    ])
    return cubes


@pytest.fixture
def cubes_2d():
    """2D cube."""
    time_coord = DimCoord(
        0.0,
        var_name='time',
        long_name='time',
        units=Unit('day since 1950-01-01 00:00:00', calendar='gregorian'),
    )
    lat_coord = DimCoord(
        0.0,
        var_name='lat',
        long_name='latitude',
        units='degrees_north',
    )
    lon_coord = DimCoord(
        0.0,
        var_name='lon',
        long_name='longitude',
        units='degrees_east',
    )
    cube = Cube(
        [[[1.0]]],
        dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )
    cubes = CubeList([
        cube.copy(),
        cube.copy(),
        cube.copy(),
        cube.copy(),
    ])
    return cubes


@pytest.fixture
def cubes_3d():
    """3D cube."""
    time_coord = DimCoord(
        0.0,
        var_name='time',
        long_name='time',
        units=Unit('day since 1950-01-01 00:00:00', calendar='gregorian'),
    )
    plev_coord = DimCoord(
        [100000.0, 90000.0],
        var_name='pax_2',
        units='Pa',
        attributes={'positive': 'down'},
    )
    lev_coord = AuxCoord(
        [1, 2],
        var_name='lev',
        long_name='hybrid level at layer midpoints',
    )
    lat_coord = DimCoord(
        0.0,
        var_name='lat',
        long_name='latitude',
        units='degrees_north',
    )
    lon_coord = DimCoord(
        0.0,
        var_name='lon',
        long_name='longitude',
        units='degrees_east',
    )
    cube = Cube(
        [[[[1.0]], [[2.0]]]],
        dim_coords_and_dims=[(time_coord, 0),
                             (plev_coord, 1),
                             (lat_coord, 2),
                             (lon_coord, 3)],
        aux_coords_and_dims=[(lev_coord, 1)],
    )
    hyam_cube = Cube(
        [100000.0, 90000.0],
        var_name='hyam',
        long_name='hybrid A coefficient at layer midpoints',
        units='Pa',
    )
    hybm_cube = Cube(
        [0.8, 0.4],
        var_name='hybm',
        long_name='hybrid B coefficient at layer midpoints',
        units='1',
    )
    hyai_cube = Cube(
        [110000.0, 95000.0, 80000.0],
        var_name='hyai',
        long_name='hybrid A coefficient at layer interfaces',
        units='Pa',
    )
    hybi_cube = Cube(
        [0.9, 0.5, 0.2],
        var_name='hybi',
        long_name='hybrid B coefficient at layer interfaces',
        units='1',
    )
    aps_ave_cube = Cube(
        [[[100000.0]]],
        var_name='aps_ave',
        long_name='surface pressure',
        units='Pa',
    )
    cubes = CubeList([
        cube.copy(),
        cube.copy(),
        cube.copy(),
        cube.copy(),
        hyam_cube,
        hybm_cube,
        hyai_cube,
        hybi_cube,
        aps_ave_cube,
    ])
    return cubes


def get_allvars_fix(mip, short_name):
    """Get member of fix class."""
    vardef = get_var_info('EMAC', mip, short_name)
    extra_facets = get_extra_facets('EMAC', 'EMAC', mip, short_name, ())
    fix = AllVars(vardef, extra_facets=extra_facets)
    return fix


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


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('day since 1950-01-01 00:00:00',
                              calendar='gregorian')
    np.testing.assert_allclose(time.points, [54786.9916666667])
    assert time.bounds is None
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
        [3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700,
         1000, 1500, 2000, 3000, 5000, 7000, 8000, 9000, 10000, 11500, 13000,
         15000, 17000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000,
         92500, 100000],
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
        [0.996141, 0.982633, 0.954782, 0.909258],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        lev.bounds[:4],
        [[1.0, 0.992281],
         [0.992281, 0.972985],
         [0.972985, 0.936579],
         [0.936579, 0.881937]],
        rtol=1e-5,
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
        [0.0, 0.0, 391.597504, 1666.582031],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        ap_coord.bounds[:4],
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 783.195007],
         [783.195007, 2549.968994]],
        rtol=1e-5,
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
        [0.996141, 0.982633, 0.950866, 0.892592],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        b_coord.bounds[:4],
        [[1.0, 0.992281],
         [0.992281, 0.972985],
         [0.972985, 0.928747],
         [0.928747, 0.856438]],
        rtol=1e-5,
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
        [[99915.351562, 98339.820312, 99585.25, 96572.765625]],
        rtol=1e-5,
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


def check_hybrid_z(cube):
    """Check hybrid Z-coordinates of 3D cubes."""
    assert len(cube.aux_factories) == 1

    air_pressure_coord = cube.coord('air_pressure')
    np.testing.assert_allclose(
        air_pressure_coord.points,
        [[[[130000.0]], [[180000.0]]]],
    )
    np.testing.assert_allclose(
        air_pressure_coord.bounds,
        [[[[[100000.0, 145000.0]]], [[[145000.0, 200000.0]]]]],
    )

    lev_coord = cube.coord('atmosphere_hybrid_sigma_pressure_coordinate')
    np.testing.assert_allclose(lev_coord.points, [1.3, 1.8])
    np.testing.assert_allclose(lev_coord.bounds, [[1.0, 1.45], [1.45, 2.0]])


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


# Test variable extraction


def test_get_cube_cav():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    cubes = CubeList([
        Cube(0.0),
        Cube(0.0, var_name='temp2_cav'),
    ])
    cube = fix.get_cube(cubes)
    assert cube.var_name == 'temp2_cav'


def test_get_cube_ave():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    cubes = CubeList([
        Cube(0.0),
        Cube(0.0, var_name='temp2_ave'),
    ])
    cube = fix.get_cube(cubes)
    assert cube.var_name == 'temp2_ave'


def test_get_cube_cav_ave():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    cubes = CubeList([
        Cube(0.0, var_name='temp2_ave'),
        Cube(0.0, var_name='temp2_cav'),
    ])
    cube = fix.get_cube(cubes)
    assert cube.var_name == 'temp2_cav'


def test_get_cube_str_input():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    cubes = CubeList([
        Cube(0.0),
        Cube(0.0, var_name='x'),
    ])
    cube = fix.get_cube(cubes, var_name='x')
    assert cube.var_name == 'x'


def test_get_cube_list_input():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    cubes = CubeList([
        Cube(0.0),
        Cube(0.0, var_name='x'),
        Cube(0.0, var_name='y'),
    ])
    cube = fix.get_cube(cubes, var_name=['y', 'x'])
    assert cube.var_name == 'y'


def test_var_not_available_fix():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cubes = CubeList([Cube(0.0)])
    msg = (r"No variable of \['tm1_p19_cav', 'tm1_p19_ave'\] necessary for "
           r"the extraction/derivation the CMOR variable 'ta' is available in "
           r"the input file.")
    with pytest.raises(ValueError, match=msg):
        fix.fix_metadata(cubes)


def test_var_not_available_get_cube():
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    cubes = CubeList([Cube(0.0)])
    msg = (r"No variable of \['x'\] necessary for the extraction/derivation "
           r"the CMOR variable 'ta' is available in the input file.")
    with pytest.raises(ValueError, match=msg):
        fix.get_cube(cubes, var_name='x')


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


# Tests with sample data
# Note: test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


def test_sample_data_tas(test_data_path, tmp_path):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    filepath = test_data_path / 'emac.nc'
    fixed_path = fix.fix_file(filepath, tmp_path)
    assert fixed_path == filepath

    cubes = iris.load(str(fixed_path))
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_tas_metadata(fixed_cubes)

    check_time(cube)
    check_lat(cube)
    check_lon(cube)

    assert cube.shape == (1, 4, 8)
    np.testing.assert_allclose(
        cube.data[:, :, 0],
        [[277.3045, 293.08575, 295.9718, 275.26523]],
        rtol=1e-5,
    )


def test_sample_data_ta_plev(test_data_path, tmp_path):
    """Test fix."""
    # Note: raw_name needs to be modified since the sample file only contains
    # plev39, while Amon's ta needs plev19 by default
    vardef = get_var_info('EMAC', 'Amon', 'ta')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'ta', ())
    original_raw_name = extra_facets['raw_name']
    extra_facets['raw_name'] = ['tm1_p39_cav', 'tm1_p39_ave']
    fix = AllVars(vardef, extra_facets=extra_facets)

    filepath = test_data_path / 'emac.nc'
    fixed_path = fix.fix_file(filepath, tmp_path)
    assert fixed_path == filepath

    cubes = iris.load(str(fixed_path))
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_ta_metadata(fixed_cubes)

    check_time(cube)
    check_plev(cube)
    check_lat(cube)
    check_lon(cube)

    assert cube.shape == (1, 39, 4, 8)
    np.testing.assert_allclose(
        cube.data[0, :5, 0, 0],
        [204.34882, 209.85188, 215.6242, 223.81247, 232.94002],
        rtol=1e-5,
    )

    fix.extra_facets['raw_name'] = original_raw_name


def test_sample_data_ta_alevel(test_data_path, tmp_path):
    """Test fix."""
    fix = get_allvars_fix('CFmon', 'ta')

    filepath = test_data_path / 'emac.nc'
    fixed_path = fix.fix_file(filepath, tmp_path)
    assert fixed_path != filepath

    cubes = iris.load(str(fixed_path))
    assert cubes.extract(NameConstraint(var_name='hyam'))
    assert cubes.extract(NameConstraint(var_name='hybm'))
    assert cubes.extract(NameConstraint(var_name='hyai'))
    assert cubes.extract(NameConstraint(var_name='hybi'))

    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_ta_metadata(fixed_cubes)

    check_time(cube)
    check_alevel(cube)
    check_lat(cube)
    check_lon(cube)

    assert cube.shape == (1, 90, 4, 8)
    np.testing.assert_allclose(
        cube.data[0, :5, 0, 0],
        [276.98267, 276.10773, 275.07455, 273.53384, 270.64545],
        rtol=1e-5,
    )


# Test 2D variables in extra_facets/emac-mappings.yml


def test_get_awhea_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Omon', 'awhea')
    assert fix == [AllVars(None)]


def test_awhea_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'awhea_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Omon', 'awhea')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'awhea'
    assert cube.standard_name is None
    assert cube.long_name == ('Global Mean Net Surface Heat Flux Over Open '
                              'Water')
    assert cube.units == 'W m-2'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_clivi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clivi')
    assert fix == [AllVars(None)]


def test_clivi_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'xivi_cav'
    cubes_2d[0].units = 'kg m-2'
    fix = get_allvars_fix('Amon', 'clivi')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clivi'
    assert cube.standard_name == 'atmosphere_mass_content_of_cloud_ice'
    assert cube.long_name == 'Ice Water Path'
    assert cube.units == 'kg m-2'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_clt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clt')
    assert fix == [Clt(None), AllVars(None)]


def test_clt_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aclcov_cav'
    vardef = get_var_info('EMAC', 'Amon', 'clt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'clt', ())
    fix = Clt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'clt')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clt'
    assert cube.standard_name == 'cloud_area_fraction'
    assert cube.long_name == 'Total Cloud Cover Percentage'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[100.0]]])


def test_get_clwvi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clwvi')
    assert fix == [Clwvi(None), AllVars(None)]


def test_clwvi_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'xlvi_cav'
    cubes_2d[1].var_name = 'xivi_cav'
    cubes_2d[0].units = 'kg m-2'
    cubes_2d[1].units = 'kg m-2'
    vardef = get_var_info('EMAC', 'Amon', 'clwvi')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'clwvi', ())
    fix = Clwvi(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[2.0]]])


def test_get_co2mass_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'co2mass')
    assert fix == [AllVars(None)]


def test_co2mass_fix(cubes_1d):
    """Test fix."""
    cubes_1d[0].var_name = 'MP_CO2_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('Amon', 'co2mass')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'co2mass'
    assert cube.standard_name == 'atmosphere_mass_of_carbon_dioxide'
    assert cube.long_name == 'Total Atmospheric Mass of CO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_evspsbl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def test_evspsbl_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'evap_cav'
    cubes_2d[0].units = 'kg m-2 s-1'
    fix = get_allvars_fix('Amon', 'evspsbl')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_hfls_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hfls')
    assert fix == [Hfls(None), AllVars(None)]


def test_hfls_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'ahfl_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'hfls')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_hfss_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hfss')
    assert fix == [Hfss(None), AllVars(None)]


def test_hfss_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'ahfs_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'hfss')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_hurs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hurs')
    assert fix == [Hurs(None), AllVars(None)]


def test_hurs_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'rh_2m_cav'
    vardef = get_var_info('EMAC', 'Amon', 'hurs')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'hurs', ())
    fix = Hurs(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'hurs')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'hurs'
    assert cube.standard_name == 'relative_humidity'
    assert cube.long_name == 'Near-Surface Relative Humidity'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[100.0]]])


def test_get_od550aer_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'od550aer')
    assert fix == [Od550aer(None), AllVars(None)]


def test_od550aer_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'aot_opt_TOT_550_total_cav'
    vardef = get_var_info('EMAC', 'Amon', 'od550aer')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'od550aer', ())
    fix = Od550aer(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_3d)

    allvars_fix = get_allvars_fix('Amon', 'od550aer')
    fixed_cubes = allvars_fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'od550aer'
    assert cube.standard_name == ('atmosphere_optical_thickness_due_to_'
                                  'ambient_aerosol_particles')
    assert cube.long_name == 'Ambient Aerosol Optical Thickness at 550nm'
    assert cube.units == '1'
    assert 'positive' not in cube.attributes

    check_lambda550nm(cube)

    np.testing.assert_allclose(cube.data, [[[3.0]]])


def test_get_pr_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'pr')
    assert fix == [Pr(None), AllVars(None)]


def test_pr_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aprl_cav'
    cubes_2d[1].var_name = 'aprc_cav'
    cubes_2d[0].units = 'kg m-2 s-1'
    cubes_2d[1].units = 'kg m-2 s-1'
    vardef = get_var_info('EMAC', 'Amon', 'pr')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'pr', ())
    fix = Pr(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'pr')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'pr'
    assert cube.standard_name == 'precipitation_flux'
    assert cube.long_name == 'Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[2.0]]])


def test_get_prc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prc')
    assert fix == [AllVars(None)]


def test_prc_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aprc_cav'
    cubes_2d[0].units = 'kg m-2 s-1'
    fix = get_allvars_fix('Amon', 'prc')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prc'
    assert cube.standard_name == 'convective_precipitation_flux'
    assert cube.long_name == 'Convective Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_prl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prl')
    assert fix == [AllVars(None)]


def test_prl_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aprl_cav'
    cubes_2d[0].units = 'kg m-2 s-1'
    fix = get_allvars_fix('Amon', 'prl')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prl'
    assert cube.standard_name is None
    assert cube.long_name == 'Large Scale Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_prsn_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prsn')
    assert fix == [AllVars(None)]


def test_prsn_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aprs_cav'
    cubes_2d[0].units = 'kg m-2 s-1'
    fix = get_allvars_fix('Amon', 'prsn')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prsn'
    assert cube.standard_name == 'snowfall_flux'
    assert cube.long_name == 'Snowfall Flux'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_prw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'prw')
    assert fix == [AllVars(None)]


def test_prw_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'qvi_cav'
    cubes_2d[0].units = 'kg m-2'
    fix = get_allvars_fix('Amon', 'prw')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'prw'
    assert cube.standard_name == 'atmosphere_mass_content_of_water_vapor'
    assert cube.long_name == 'Water Vapor Path'
    assert cube.units == 'kg m-2'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_ps_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ps')
    assert fix == [AllVars(None)]


def test_ps_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'aps_cav'
    cubes_2d[0].units = 'Pa'
    fix = get_allvars_fix('Amon', 'ps')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ps'
    assert cube.standard_name == 'surface_air_pressure'
    assert cube.long_name == 'Surface Air Pressure'
    assert cube.units == 'Pa'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_psl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'psl')
    assert fix == [AllVars(None)]


def test_psl_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'slp_cav'
    cubes_2d[0].units = 'Pa'
    fix = get_allvars_fix('Amon', 'psl')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'psl'
    assert cube.standard_name == 'air_pressure_at_mean_sea_level'
    assert cube.long_name == 'Sea Level Pressure'
    assert cube.units == 'Pa'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_rlds_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlds')
    assert fix == [Rlds(None), AllVars(None)]


def test_rlds_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxtbot_cav'
    cubes_2d[1].var_name = 'tradsu_cav'
    cubes_2d[0].units = 'W m-2'
    cubes_2d[1].units = 'W m-2'
    vardef = get_var_info('EMAC', 'Amon', 'rlds')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rlds', ())
    fix = Rlds(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'rlds')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rlds'
    assert cube.standard_name == 'surface_downwelling_longwave_flux_in_air'
    assert cube.long_name == 'Surface Downwelling Longwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[0.0]]])


def test_get_rlus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlus')
    assert fix == [Rlus(None), AllVars(None)]


def test_rlus_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'tradsu_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rlus')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rlut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlut')
    assert fix == [Rlut(None), AllVars(None)]


def test_rlut_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxttop_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rlut')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rlutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rlutcs')
    assert fix == [Rlutcs(None), AllVars(None)]


def test_rlutcs_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxtftop_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rlutcs')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rsds_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsds')
    assert fix == [Rsds(None), AllVars(None)]


def test_rsds_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxsbot_cav'
    cubes_2d[1].var_name = 'sradsu_cav'
    cubes_2d[0].units = 'W m-2'
    cubes_2d[1].units = 'W m-2'
    vardef = get_var_info('EMAC', 'Amon', 'rsds')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsds', ())
    fix = Rsds(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'rsds')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsds'
    assert cube.standard_name == 'surface_downwelling_shortwave_flux_in_air'
    assert cube.long_name == 'Surface Downwelling Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[0.0]]])


def test_get_rsdt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsdt')
    assert fix == [Rsdt(None), AllVars(None)]


def test_rsdt_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxstop_cav'
    cubes_2d[1].var_name = 'srad0u_cav'
    cubes_2d[0].units = 'W m-2'
    cubes_2d[1].units = 'W m-2'
    vardef = get_var_info('EMAC', 'Amon', 'rsdt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rsdt', ())
    fix = Rsdt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('Amon', 'rsdt')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'rsdt'
    assert cube.standard_name == 'toa_incoming_shortwave_flux'
    assert cube.long_name == 'TOA Incident Shortwave Radiation'
    assert cube.units == 'W m-2'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[0.0]]])


def test_get_rsus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsus')
    assert fix == [Rsus(None), AllVars(None)]


def test_rsus_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'sradsu_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rsus')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rsut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsut')
    assert fix == [Rsut(None), AllVars(None)]


def test_rsut_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'srad0u_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rsut')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rsutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rsutcs')
    assert fix == [Rsutcs(None), AllVars(None)]


def test_rsutcs_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxusftop_cav'
    cubes_2d[0].units = 'W m-2'
    fix = get_allvars_fix('Amon', 'rsutcs')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[-1.0]]])


def test_get_rtmt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'rtmt')
    assert fix == [Rtmt(None), AllVars(None)]


def test_rtmt_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'flxttop_cav'
    cubes_2d[1].var_name = 'flxstop_cav'
    cubes_2d[0].units = 'W m-2'
    cubes_2d[1].units = 'W m-2'
    vardef = get_var_info('EMAC', 'Amon', 'rtmt')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'rtmt', ())
    fix = Rtmt(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[2.0]]])


def test_get_sfcWind_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'sfcWind')
    assert fix == [AllVars(None)]


def test_sfcWind_fix(cubes_2d):  # noqa: N802
    """Test fix."""
    cubes_2d[0].var_name = 'wind10_cav'
    cubes_2d[0].units = 'm s-1'
    fix = get_allvars_fix('Amon', 'sfcWind')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'sfcWind'
    assert cube.standard_name == 'wind_speed'
    assert cube.long_name == 'Near-Surface Wind Speed'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 10.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconc')
    assert fix == [Siconc(None), AllVars(None)]


def test_siconc_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'seaice_cav'
    vardef = get_var_info('EMAC', 'SImon', 'siconc')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconc', ())
    fix = Siconc(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('SImon', 'siconc')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'siconc'
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == 'Sea-Ice Area Percentage (Ocean Grid)'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    check_typesi(cube)

    np.testing.assert_allclose(cube.data, [[[100.0]]])


def test_get_siconca_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'siconca')
    assert fix == [Siconca(None), AllVars(None)]


def test_siconca_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'seaice_cav'
    vardef = get_var_info('EMAC', 'SImon', 'siconca')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'SImon', 'siconca', ())
    fix = Siconca(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fix = get_allvars_fix('SImon', 'siconca')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'siconca'
    assert cube.standard_name == 'sea_ice_area_fraction'
    assert cube.long_name == 'Sea-Ice Area Percentage (Atmospheric Grid)'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    check_typesi(cube)

    np.testing.assert_allclose(cube.data, [[[100.0]]])


def test_get_sithick_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'SImon', 'sithick')
    assert fix == [Sithick(None), AllVars(None)]


def test_sithick_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'siced_cav'
    cubes_2d[0].units = 'm'
    fix = get_allvars_fix('SImon', 'sithick')
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[1.0]]])
    np.testing.assert_equal(np.ma.getmaskarray(cube.data), [[[False]]])

    # Check masking
    cube.data = [[[0.0]]]
    cube = fix.fix_data(cube)
    np.testing.assert_allclose(cube.data, [[[0.0]]])
    np.testing.assert_equal(np.ma.getmaskarray(cube.data), [[[True]]])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tas')
    assert fix == [AllVars(None)]


def test_tas_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'temp2_cav'
    cubes_2d[0].units = 'K'
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 2.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_tasmax_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tasmax')
    assert fix == [AllVars(None)]


def test_tasmax_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'temp2_max'
    cubes_2d[0].units = 'K'
    fix = get_allvars_fix('Amon', 'tasmax')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tasmax'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Daily Maximum Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 2.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_tasmin_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tasmin')
    assert fix == [AllVars(None)]


def test_tasmin_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'temp2_min'
    cubes_2d[0].units = 'K'
    fix = get_allvars_fix('Amon', 'tasmin')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tasmin'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Daily Minimum Near-Surface Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 2.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_tauu_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tauu')
    assert fix == [AllVars(None)]


def test_tauu_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'ustr_cav'
    cubes_2d[0].units = 'Pa'
    fix = get_allvars_fix('Amon', 'tauu')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tauu'
    assert cube.standard_name == 'surface_downward_eastward_stress'
    assert cube.long_name == 'Surface Downward Eastward Wind Stress'
    assert cube.units == 'Pa'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_tauv_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'tauv')
    assert fix == [AllVars(None)]


def test_tauv_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'vstr_cav'
    cubes_2d[0].units = 'Pa'
    fix = get_allvars_fix('Amon', 'tauv')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tauv'
    assert cube.standard_name == 'surface_downward_northward_stress'
    assert cube.long_name == 'Surface Downward Northward Wind Stress'
    assert cube.units == 'Pa'
    assert cube.attributes['positive'] == 'down'

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Omon', 'tos')
    assert fix == [AllVars(None)]


def test_tos_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'tsw'
    cubes_2d[0].units = 'degC'
    fix = get_allvars_fix('Omon', 'tos')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'tos'
    assert cube.standard_name == 'sea_surface_temperature'
    assert cube.long_name == 'Sea Surface Temperature'
    assert cube.units == 'degC'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_toz_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'AERmon', 'toz')
    assert fix == [Toz(None), AllVars(None)]


def test_toz_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'toz'
    cubes_2d[0].units = 'DU'
    vardef = get_var_info('EMAC', 'AERmon', 'toz')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'AERmon', 'toz', ())
    fix = Toz(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_2d)

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

    np.testing.assert_allclose(cube.data, [[[1e-5]]])


def test_get_ts_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ts')
    assert fix == [AllVars(None)]


def test_ts_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'tsurf_cav'
    cubes_2d[0].units = 'K'
    fix = get_allvars_fix('Amon', 'ts')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ts'
    assert cube.standard_name == 'surface_temperature'
    assert cube.long_name == 'Surface Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_uas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'uas')
    assert fix == [AllVars(None)]


def test_uas_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'u10_cav'
    cubes_2d[0].units = 'm s-1'
    fix = get_allvars_fix('Amon', 'uas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'uas'
    assert cube.standard_name == 'eastward_wind'
    assert cube.long_name == 'Eastward Near-Surface Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 10.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


def test_get_vas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'vas')
    assert fix == [AllVars(None)]


def test_vas_fix(cubes_2d):
    """Test fix."""
    cubes_2d[0].var_name = 'v10_cav'
    cubes_2d[0].units = 'm s-1'
    fix = get_allvars_fix('Amon', 'vas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'vas'
    assert cube.standard_name == 'northward_wind'
    assert cube.long_name == 'Northward Near-Surface Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    check_heightxm(cube, 10.0)

    np.testing.assert_allclose(cube.data, [[[1.0]]])


# Test 1D tracers in extra_facets/emac-mappings.yml


def test_get_MP_BC_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_BC_tot')
    assert fix == [MP_BC_tot(None), AllVars(None)]


def test_MP_BC_tot_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_BC_ki_cav'
    cubes_1d[1].var_name = 'MP_BC_ks_cav'
    cubes_1d[2].var_name = 'MP_BC_as_cav'
    cubes_1d[3].var_name = 'MP_BC_cs_cav'
    cubes_1d[0].units = 'kg'
    cubes_1d[1].units = 'kg'
    cubes_1d[2].units = 'kg'
    cubes_1d[3].units = 'kg'
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_BC_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_BC_tot',
                                    ())
    fix = MP_BC_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_1d)

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

    np.testing.assert_allclose(cube.data, [4.0])


def test_get_MP_CFCl3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CFCl3')
    assert fix == [AllVars(None)]


def test_MP_CFCl3_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_CFCl3_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_CFCl3')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CFCl3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CFCl3 (CFC-11)'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_ClOX_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_ClOX')
    assert fix == [AllVars(None)]


def test_MP_ClOX_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_ClOX_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_ClOX')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_ClOX'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of ClOX'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_CH4_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CH4')
    assert fix == [AllVars(None)]


def test_MP_CH4_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_CH4_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_CH4')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CH4'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CH4'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_CO_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CO')
    assert fix == [AllVars(None)]


def test_MP_CO_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_CO_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_CO')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CO'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CO'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_CO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_CO2')
    assert fix == [AllVars(None)]


def test_MP_CO2_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_CO2_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_CO2')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_CO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of CO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_DU_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_DU_tot')
    assert fix == [MP_DU_tot(None), AllVars(None)]


def test_MP_DU_tot_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_DU_ai_cav'
    cubes_1d[1].var_name = 'MP_DU_as_cav'
    cubes_1d[2].var_name = 'MP_DU_ci_cav'
    cubes_1d[3].var_name = 'MP_DU_cs_cav'
    cubes_1d[0].units = 'kg'
    cubes_1d[1].units = 'kg'
    cubes_1d[2].units = 'kg'
    cubes_1d[3].units = 'kg'
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_DU_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_DU_tot',
                                    ())
    fix = MP_DU_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_1d)

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

    np.testing.assert_allclose(cube.data, [4.0])


def test_get_MP_N2O_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_N2O')
    assert fix == [AllVars(None)]


def test_MP_N2O_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_N2O_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_N2O')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_N2O'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of N2O'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_NH3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NH3')
    assert fix == [AllVars(None)]


def test_MP_NH3_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_NH3_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_NH3')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NH3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NH3'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_NO_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NO')
    assert fix == [AllVars(None)]


def test_MP_NO_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_NO_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_NO')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NO'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NO'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_NO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NO2')
    assert fix == [AllVars(None)]


def test_MP_NO2_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_NO2_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_NO2')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_NOX_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_NOX')
    assert fix == [AllVars(None)]


def test_MP_NOX_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_NOX_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_NOX')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_NOX'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of NOX (NO+NO2)'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_O3_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_O3')
    assert fix == [AllVars(None)]


def test_MP_O3_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_O3_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_O3')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_O3'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of O3'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_OH_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_OH')
    assert fix == [AllVars(None)]


def test_MP_OH_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_OH_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_OH')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_OH'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of OH'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_S_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_S')
    assert fix == [AllVars(None)]


def test_MP_S_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_S_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_S')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_S'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of S'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_SO2_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO2')
    assert fix == [AllVars(None)]


def test_MP_SO2_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_SO2_cav'
    cubes_1d[0].units = 'kg'
    fix = get_allvars_fix('TRAC10hr', 'MP_SO2')
    fixed_cubes = fix.fix_metadata(cubes_1d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'MP_SO2'
    assert cube.standard_name is None
    assert cube.long_name == 'total mass of SO2'
    assert cube.units == 'kg'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, [1.0])


def test_get_MP_SO4mm_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO4mm_tot')
    assert fix == [MP_SO4mm_tot(None), AllVars(None)]


def test_MP_SO4mm_tot_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_SO4mm_ns_cav'
    cubes_1d[1].var_name = 'MP_SO4mm_ks_cav'
    cubes_1d[2].var_name = 'MP_SO4mm_as_cav'
    cubes_1d[3].var_name = 'MP_SO4mm_cs_cav'
    cubes_1d[0].units = 'kg'
    cubes_1d[1].units = 'kg'
    cubes_1d[2].units = 'kg'
    cubes_1d[3].units = 'kg'
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_SO4mm_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_SO4mm_tot',
                                    ())
    fix = MP_SO4mm_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_1d)

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

    np.testing.assert_allclose(cube.data, [4.0])


def test_get_MP_SS_tot_fix():  # noqa: N802
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'TRAC10hr', 'MP_SS_tot')
    assert fix == [MP_SS_tot(None), AllVars(None)]


def test_MP_SS_tot_fix(cubes_1d):  # noqa: N802
    """Test fix."""
    cubes_1d[0].var_name = 'MP_SS_ks_cav'
    cubes_1d[1].var_name = 'MP_SS_as_cav'
    cubes_1d[2].var_name = 'MP_SS_cs_cav'
    cubes_1d[0].units = 'kg'
    cubes_1d[1].units = 'kg'
    cubes_1d[2].units = 'kg'
    vardef = get_var_info('EMAC', 'TRAC10hr', 'MP_SS_tot')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'TRAC10hr', 'MP_SS_tot',
                                    ())
    fix = MP_SS_tot(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_1d)

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

    np.testing.assert_allclose(cube.data, [3.0])


# Test 3D variables in extra_facets/emac-mappings.yml


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


def test_cl_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'aclcac_cav'
    vardef = get_var_info('EMAC', 'Amon', 'cl')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'cl', ())
    fix = Cl(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_3d)

    fix = get_allvars_fix('Amon', 'cl')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'cl'
    assert cube.standard_name == 'cloud_area_fraction_in_atmosphere_layer'
    assert cube.long_name == 'Percentage Cloud Cover'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    check_hybrid_z(cube)

    np.testing.assert_allclose(cube.data, [[[[200.0]], [[100.0]]]])


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'cli')
    assert fix == [AllVars(None)]


def test_cli_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'xim1_cav'
    cubes_3d[0].units = 'kg kg-1'
    fix = get_allvars_fix('Amon', 'cli')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'cli'
    assert cube.standard_name == 'mass_fraction_of_cloud_ice_in_air'
    assert cube.long_name == 'Mass Fraction of Cloud Ice'
    assert cube.units == 'kg kg-1'
    assert 'positive' not in cube.attributes

    check_hybrid_z(cube)

    np.testing.assert_allclose(cube.data, [[[[2.0]], [[1.0]]]])


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'clw')
    assert fix == [AllVars(None)]


def test_clw_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'xlm1_cav'
    cubes_3d[0].units = 'kg kg-1'
    fix = get_allvars_fix('Amon', 'clw')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'clw'
    assert cube.standard_name == 'mass_fraction_of_cloud_liquid_water_in_air'
    assert cube.long_name == 'Mass Fraction of Cloud Liquid Water'
    assert cube.units == 'kg kg-1'
    assert 'positive' not in cube.attributes

    check_hybrid_z(cube)

    np.testing.assert_allclose(cube.data, [[[[2.0]], [[1.0]]]])


def test_get_hur_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hur')
    assert fix == [AllVars(None)]


def test_hur_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'rhum_p19_cav'
    cubes_3d[0].units = '1'
    fix = get_allvars_fix('Amon', 'hur')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'hur'
    assert cube.standard_name == 'relative_humidity'
    assert cube.long_name == 'Relative Humidity'
    assert cube.units == '%'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(cube.data, [[[[100.0]], [[200.0]]]])


def test_get_hus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'hus')
    assert fix == [AllVars(None)]


def test_hus_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'qm1_p19_cav'
    cubes_3d[0].units = '1'
    fix = get_allvars_fix('Amon', 'hus')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'hus'
    assert cube.standard_name == 'specific_humidity'
    assert cube.long_name == 'Specific Humidity'
    assert cube.units == '1'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(cube.data, [[[[1.0]], [[2.0]]]])


def test_get_ta_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ta')
    assert fix == [AllVars(None)]


def test_ta_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'tm1_p19_cav'
    cubes_3d[0].units = 'K'
    fix = get_allvars_fix('Amon', 'ta')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ta'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Air Temperature'
    assert cube.units == 'K'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(cube.data, [[[[1.0]], [[2.0]]]])


def test_get_ua_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'ua')
    assert fix == [AllVars(None)]


def test_ua_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'um1_p19_cav'
    cubes_3d[0].units = 'm s-1'
    fix = get_allvars_fix('Amon', 'ua')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'ua'
    assert cube.standard_name == 'eastward_wind'
    assert cube.long_name == 'Eastward Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(cube.data, [[[[1.0]], [[2.0]]]])


def test_get_va_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'va')
    assert fix == [AllVars(None)]


def test_va_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'vm1_p19_cav'
    cubes_3d[0].units = 'm s-1'
    fix = get_allvars_fix('Amon', 'va')
    fixed_cubes = fix.fix_metadata(cubes_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'va'
    assert cube.standard_name == 'northward_wind'
    assert cube.long_name == 'Northward Wind'
    assert cube.units == 'm s-1'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(cube.data, [[[[1.0]], [[2.0]]]])


def test_get_zg_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('EMAC', 'EMAC', 'Amon', 'zg')
    assert fix == [Zg(None), AllVars(None)]


def test_zg_fix(cubes_3d):
    """Test fix."""
    cubes_3d[0].var_name = 'geopot_p19_cav'
    cubes_3d[0].units = 'm2 s-2'
    vardef = get_var_info('EMAC', 'Amon', 'zg')
    extra_facets = get_extra_facets('EMAC', 'EMAC', 'Amon', 'zg', ())
    fix = Zg(vardef, extra_facets=extra_facets)
    fixed_cubes = fix.fix_metadata(cubes_3d)

    fix = get_allvars_fix('Amon', 'zg')
    fixed_cubes = fix.fix_metadata(fixed_cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == 'zg'
    assert cube.standard_name == 'geopotential_height'
    assert cube.long_name == 'Geopotential Height'
    assert cube.units == 'm'
    assert 'positive' not in cube.attributes

    assert not cube.aux_factories
    assert cube.coords('air_pressure')

    np.testing.assert_allclose(
        cube.data,
        [[[[0.101971]], [[0.203943]]]],
        rtol=1e-5,
    )


# Test ``AllVars.fix_file``


@mock.patch('esmvalcore.cmor._fixes.emac.emac.copyfile', autospec=True)
def test_fix_file_no_alevel(mock_copyfile):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'ta')
    new_path = fix.fix_file(mock.sentinel.filepath, mock.sentinel.output_dir)

    assert new_path == mock.sentinel.filepath
    mock_copyfile.assert_not_called()


# Test ``AllVars._fix_plev``


def test_fix_plev_no_plev_coord(cubes_3d):
    """Test fix."""
    # Create cube with Z-coord whose units are not convertible to Pa
    cube = cubes_3d[0]
    z_coord = cube.coord(axis='Z')
    z_coord.var_name = 'height'
    z_coord.standard_name = 'height'
    z_coord.long_name = 'height'
    z_coord.units = 'm'
    z_coord.attributes = {'positive': 'up'}
    z_coord.points = np.arange(z_coord.shape[0])[::-1]

    fix = get_allvars_fix('Amon', 'ta')

    msg = ("Cannot find requested pressure level coordinate for variable "
           "'ta', searched for Z-coordinates with units that are convertible "
           "to Pa")
    with pytest.raises(ValueError, match=msg):
        fix._fix_plev(cube)


# Test fix invalid units (using INVALID_UNITS)


def test_fix_invalid_units():
    """Test fix."""
    cube = Cube(1.0, attributes={'invalid_units': 'kg/m**2s'})

    fix = get_allvars_fix('Amon', 'pr')
    fix.fix_var_metadata(cube)

    assert cube.var_name == 'pr'
    assert cube.standard_name == 'precipitation_flux'
    assert cube.long_name == 'Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert cube.units.origin == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, 1.0)
