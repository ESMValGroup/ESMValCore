"""Tests for the ACCESS-ESM on-the-fly CMORizer."""

import iris
import numpy as np
import pytest

import esmvalcore.cmor._fixes.access.access_esm
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CoordinateInfo, get_var_info
from esmvalcore.config._config import get_extra_facets
from esmvalcore.dataset import Dataset


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / 'access_native.nc'
    return iris.load(str(nc_path))


def _get_fix(mip, frequency, short_name, fix_name):
    """Load a fix from :mod:`esmvalcore.cmor._fixes.cesm.cesm2`."""
    dataset = Dataset(
        project='ACCESS',
        dataset='ACCESS_ESM',
        mip=mip,
        short_name=short_name,
    )
    extra_facets = get_extra_facets(dataset, ())
    extra_facets['frequency'] = frequency
    extra_facets['exp'] = 'amip'
    vardef = get_var_info(project='ACCESS', mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.access.access_esm, fix_name)
    fix = cls(vardef, extra_facets=extra_facets, session={}, frequency='')
    return fix


def get_fix(mip, frequency, short_name):
    """Load a variable fix from esmvalcore.cmor._fixes.cesm.cesm."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, frequency, short_name, fix_name)


def fix_metadata(cubes, mip, frequency, short_name):
    """Fix metadata of cubes."""
    fix = get_fix(mip, frequency, short_name)
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


def check_pr_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == 'pr'
    assert cube.standard_name == 'precipitation_flux'
    assert cube.long_name == 'Precipitation'
    assert cube.units == 'kg m-2 s-1'
    assert 'positive' not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords('time', dim_coords=True)
    time = cube.coord('time', dim_coords=True)
    assert time.var_name == 'time'
    assert time.standard_name == 'time'
    # assert time.long_name == 'time'
    # assert time.units == Unit('days since 1979-01-01',
    #                           calendar='proleptic_gregorian')
    # np.testing.assert_allclose(
    #     time.points,
    #     [
    #         7649.5, 7680.5, 7710.0, 7739.5, 7770.0, 7800.5, 7831.0, 7861.5,
    #         7892.5, 7923.0, 7953.5, 7984.0
    #     ],
    # )
    assert time.bounds.shape == (1, 2)
    assert time.attributes == {}


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords('latitude', dim_coords=True)
    lat = cube.coord('latitude', dim_coords=True)
    assert lat.var_name == 'lat'
    assert lat.standard_name == 'latitude'
    # assert lat.long_name == 'latitude'
    assert lat.units == 'degrees_north'
    # np.testing.assert_allclose(
    #     lat.points,
    #     [
    #         59.4444082891668, 19.8757191474409, -19.8757191474409,
    #         -59.4444082891668
    #     ],
    # )
    # np.testing.assert_allclose(
    #     lat.bounds,
    #     [[90.0, 39.384861047478], [39.384861047478, 0.0],
    #      [0.0, -39.384861047478], [-39.384861047478, -90.0]],
    # )
    assert lat.attributes == {}


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords('longitude', dim_coords=True)
    lon = cube.coord('longitude', dim_coords=True)
    assert lon.var_name == 'lon'
    assert lon.standard_name == 'longitude'
    # assert lon.long_name == 'longitude'
    assert lon.units == 'degrees_east'
    # np.testing.assert_allclose(
    #     lon.points,
    #     [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    # )
    # np.testing.assert_allclose(
    #     lon.bounds,
    #     [[-22.5, 22.5], [22.5, 67.5], [67.5, 112.5], [112.5, 157.5],
    #      [157.5, 202.5], [202.5, 247.5], [247.5, 292.5], [292.5, 337.5]],
    # )
    assert lon.attributes == {}


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    # assert height.long_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def test_only_time(monkeypatch, cubes_2d):
    """Test fix."""
    var_list=['tas','pr']

    for var in var_list:
        fix = get_fix('Amon', 'mon', var)

        # We know that tas has dimensions time, latitude, longitude, but the CESM2
        # CMORizer is designed to check for the presence of each dimension
        # individually. To test this, remove all but one dimension of tas to create
        # an artificial, but realistic test case.
        coord_info = CoordinateInfo('time')
        coord_info.standard_name = 'time'
        monkeypatch.setattr(fix.vardef, 'coordinates', {'time': coord_info})

        cubes = cubes_2d

        # time_coord = DimCoord([0.0, 1.0], var_name='time', standard_name='time',
        #                       long_name='time', units='days since 1850-01-01')
        # height_coord = DimCoord([1.5], var_name='height_0',
        #                       standard_name='height', units='m')
        # cubes = CubeList([
        #     Cube([1, 1], var_name='fld_s03i236', units='K',
        #          dim_coords_and_dims=[(time_coord, 0)]),
        # ])
        # cubes[0].add_aux_coord(height_coord)
        fixed_cubes = fix.fix_metadata(cubes)

        # Check cube metadata
        if var == 'tas':
            cube = check_tas_metadata(fixed_cubes)
        elif var == 'pr':
            cube == check_pr_metadata(fixed_cubes)
        # cube = fixed_cubes
        # Check cube data
        assert cube.shape == (1, 145, 192)

        # Check time metadata
        assert cube.coords('time')
        new_time_coord = cube.coord('time', dim_coords=True)
        assert new_time_coord.var_name == 'time'
        assert new_time_coord.standard_name == 'time'
        # assert new_time_coord.long_name == 'time'
        # assert new_time_coord.units == Unit('days since 1979-01-01',
        #                           calendar='proleptic_gregorian')

        # # Check time data
        # np.testing.assert_allclose(new_time_coord.points, [0.0, 1.0])
        # np.testing.assert_allclose(new_time_coord.bounds,
        #                            [[-0.5, 0.5], [0.5, 1.5]])


def test_only_latitude(monkeypatch, cubes_2d):
    """Test fix."""
    var_list=['tas','pr']

    for var in var_list:
        fix = get_fix('Amon', 'mon', 'tas')

        # We know that tas has dimensions time, latitude, longitude, but the CESM2
        # CMORizer is designed to check for the presence of each dimension
        # individually. To test this, remove all but one dimension of tas to create
        # an artificial, but realistic test case.
        coord_info = CoordinateInfo('latitude')
        coord_info.standard_name = 'latitude'
        monkeypatch.setattr(fix.vardef, 'coordinates', {'latitude': coord_info})

        cubes = cubes_2d
        fixed_cubes = fix.fix_metadata(cubes)

        # Check cube metadata
        if var == 'tas':
            cube = check_tas_metadata(fixed_cubes)
        elif var == 'pr':
            cube = check_pr_metadata(fixed_cubes)

        # Check cube data
        assert cube.shape == (1, 145, 192)

        # Check latitude metadata
        assert cube.coords('latitude', dim_coords=True)
        new_lat_coord = cube.coord('latitude')
        assert new_lat_coord.var_name == 'lat'
        assert new_lat_coord.standard_name == 'latitude'
        # assert new_lat_coord.long_name == 'latitude'
        assert new_lat_coord.units == 'degrees_north'

        # Check latitude data
        # np.testing.assert_allclose(new_lat_coord.points, [0.0, 10.0])
        # np.testing.assert_allclose(new_lat_coord.bounds,
        #                            [[-5.0, 5.0], [5.0, 15.0]])


def test_only_longitude(monkeypatch, cubes_2d):
    """Test fix."""
    var_list=['tas','pr']

    for var in var_list:
        fix = get_fix('Amon', 'mon', 'tas')

        # We know that tas has dimensions time, latitude, longitude, but the CESM2
        # CMORizer is designed to check for the presence of each dimension
        # individually. To test this, remove all but one dimension of tas to create
        # an artificial, but realistic test case.
        coord_info = CoordinateInfo('longitude')
        coord_info.standard_name = 'longitude'
        monkeypatch.setattr(fix.vardef, 'coordinates', {'longitude': coord_info})

        cubes = cubes_2d
        fixed_cubes = fix.fix_metadata(cubes)

        # Check cube metadata
        if var == 'tas':
            cube = check_tas_metadata(fixed_cubes)
        elif var =='pr':
            cube = check_pr_metadata(fixed_cubes)

        # Check cube data
        assert cube.shape == (1, 145, 192)
        # np.testing.assert_equal(cube.data, [1, 1])

        # Check longitude metadata
        assert cube.coords('longitude', dim_coords=True)
        new_lon_coord = cube.coord('longitude')
        assert new_lon_coord.var_name == 'lon'
        assert new_lon_coord.standard_name == 'longitude'
        # assert new_lon_coord.long_name == 'longitude'
        assert new_lon_coord.units == 'degrees_east'

        # Check longitude data
        # np.testing.assert_allclose(new_lon_coord.points, [0.0, 180.0])
        # np.testing.assert_allclose(new_lon_coord.bounds,
        #                            [[-90.0, 90.0], [90.0, 270.0]])


def test_get_tas_fix():
    """Test getting of fix 'tas'."""
    fix = Fix.get_fixes('ACCESS', 'ACCESS_ESM', 'Amon', 'tas')
    assert fix == [
        esmvalcore.cmor._fixes.access.access_esm.Tas(vardef={},
                                                     extra_facets={},
                                                     session={},
                                                     frequency=''),
        GenericFix(None),
    ]


def test_get_tas_fix():
    """Test getting of fix 'pr'."""
    fix = Fix.get_fixes('ACCESS', 'ACCESS_ESM', 'Amon', 'pr')
    assert fix == [
        esmvalcore.cmor._fixes.access.access_esm.Pr(vardef={},
                                                     extra_facets={},
                                                     session={},
                                                     frequency=''),
        GenericFix(None),
    ]


def test_tas_fix(cubes_2d):
    """Test fix 'tas'."""
    fix = get_fix('Amon', 'mon', 'tas')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fixed_cube = check_tas_metadata(fixed_cubes)

    check_time(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)
    check_heightxm(fixed_cube, 1.5)

    assert fixed_cube.shape == (1, 145, 192)


def test_pr_fix(cubes_2d):
    """Test fix 'pr'."""
    fix = get_fix('Amon', 'mon', 'pr')
    fixed_cubes = fix.fix_metadata(cubes_2d)

    fixed_cube = check_pr_metadata(fixed_cubes)

    check_time(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)

    assert fixed_cube.shape == (1, 145, 192)
