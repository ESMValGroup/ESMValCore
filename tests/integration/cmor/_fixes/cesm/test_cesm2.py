"""Tests for the CESM2 on-the-fly CMORizer."""
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.cmor._fixes.cesm.cesm2
from esmvalcore._config import get_extra_facets
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info

# Note: test_data_path is defined in tests/integration/cmor/_fixes/conftest.py


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / 'cesm2_native.nc'
    return iris.load(str(nc_path))


def _get_fix(mip, short_name, fix_name):
    """Load a fix from :mod:`esmvalcore.cmor._fixes.cesm.cesm2`."""
    extra_facets = get_extra_facets('CESM', 'CESM2', mip, short_name, ())
    vardef = get_var_info(project='CESM', mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.cesm.cesm2, fix_name)
    fix = cls(vardef, extra_facets=extra_facets)
    return fix


def get_fix(mip, short_name):
    """Load a variable fix from esmvalcore.cmor._fixes.cesm.cesm."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, short_name, fix_name)


def get_allvars_fix(mip, short_name):
    """Load the AllVars fix from esmvalcore.cmor._fixes.cesm.cesm."""
    return _get_fix(mip, short_name, 'AllVars')


def fix_metadata(cubes, mip, short_name):
    """Fix metadata of cubes."""
    fix = get_fix(mip, short_name)
    cubes = fix.fix_metadata(cubes)
    fix = get_allvars_fix(mip, short_name)
    cubes = fix.fix_metadata(cubes)
    return cubes


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


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    assert time.long_name == 'time'
    assert time.units == Unit('days since 1979-01-01 00:00:00',
                              calendar='365_day')
    assert time.shape == (12,)
    assert time.bounds.shape == (12, 2)
    assert time.attributes == {}


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
        [[90.0, 39.384861047478],
         [39.384861047478, 0.0],
         [0.0, -39.384861047478],
         [-39.384861047478, -90.0]],
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


# Test with single-dimension cubes


def test_only_time(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # We know that tas has dimensions time, latitude, longitude, but the CESM2
    # CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of tas to create
    # an artificial, but realistic test case.
    monkeypatch.setattr(fix.vardef, 'dimensions', ['time'])

    # Create cube with only a single dimension
    time_coord = DimCoord([0.0, 1.0], var_name='time', standard_name='time',
                          long_name='time', units='days since 1850-01-01')
    cubes = CubeList([
        Cube([1, 1], var_name='TREFHT', units='K',
             dim_coords_and_dims=[(time_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

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


def test_only_latitude(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # We know that tas has dimensions time, latitude, longitude, but the CESM2
    # CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of tas to create
    # an artificial, but realistic test case.
    monkeypatch.setattr(fix.vardef, 'dimensions', ['latitude'])

    # Create cube with only a single dimension
    lat_coord = DimCoord([0.0, 10.0], var_name='lat', standard_name='latitude',
                         units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='TREFHT', units='K',
             dim_coords_and_dims=[(lat_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

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


def test_only_longitude(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # We know that tas has dimensions time, latitude, longitude, but the CESM2
    # CMORizer is designed to check for the presence of each dimension
    # individually. To test this, remove all but one dimension of tas to create
    # an artificial, but realistic test case.
    monkeypatch.setattr(fix.vardef, 'dimensions', ['longitude'])

    # Create cube with only a single dimension
    lon_coord = DimCoord([0.0, 180.0], var_name='lon',
                         standard_name='longitude', units='degrees')
    cubes = CubeList([
        Cube([1, 1], var_name='TREFHT', units='K',
             dim_coords_and_dims=[(lon_coord, 0)]),
    ])
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

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


# Test 2D variables in extra_facets/cesm-mappings.yml


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CESM', 'CESM2', 'Amon', 'tas')
    assert fix == [
        esmvalcore.cmor._fixes.cesm.cesm2.AllVars(None),
    ]


def test_tas_fix(cubes_2d):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fixed_cube = check_tas_metadata(fixed_cubes)

    check_time(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)
    check_heightxm(fixed_cube, 2.0)

    assert fixed_cube.shape == (12, 4, 8)


# Test fix invalid units (using INVALID_UNITS)


def test_fix_invalid_units(monkeypatch):
    """Test fix."""
    fix = get_allvars_fix('Amon', 'tas')

    # We know that tas has units 'K', but to check if the invalid units
    # 'fraction' are correctly handled, we change tas' units to '1'. This is an
    # artificial, but realistic test case.
    monkeypatch.setattr(fix.vardef, 'units', '1')
    cube = Cube(1.0, attributes={'invalid_units': 'fraction'})
    fix.fix_var_metadata(cube)

    assert cube.var_name == 'tas'
    assert cube.standard_name == 'air_temperature'
    assert cube.long_name == 'Near-Surface Air Temperature'
    assert cube.units == '1'
    assert 'positive' not in cube.attributes

    np.testing.assert_allclose(cube.data, 1.0)
