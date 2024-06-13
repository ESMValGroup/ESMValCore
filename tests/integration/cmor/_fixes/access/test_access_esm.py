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


def get_fix_allvar(mip, frequency, short_name):
    """Load a variable fix from esmvalcore.cmor._fixes.cesm.cesm."""
    return _get_fix(mip, frequency, short_name, 'AllVar')


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
    assert time.bounds.shape == (1, 2)
    assert time.attributes == {}


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords('latitude', dim_coords=True)
    lat = cube.coord('latitude', dim_coords=True)
    assert lat.var_name == 'lat'
    assert lat.standard_name == 'latitude'
    assert lat.units == 'degrees_north'
    assert lat.attributes == {}


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords('longitude', dim_coords=True)
    lon = cube.coord('longitude', dim_coords=True)
    assert lon.var_name == 'lon'
    assert lon.standard_name == 'longitude'
    assert lon.units == 'degrees_east'
    assert lon.attributes == {}


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords('height')
    height = cube.coord('height')
    assert height.var_name == 'height'
    assert height.standard_name == 'height'
    assert height.units == 'm'
    assert height.attributes == {'positive': 'up'}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def test_only_time(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix_allvar('Amon', 'mon', 'tas')

    coord_info = CoordinateInfo('time')
    coord_info.standard_name = 'time'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'time': coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check time metadata
    assert cube.coords('time')
    new_time_coord = cube.coord('time', dim_coords=True)
    assert new_time_coord.var_name == 'time'
    assert new_time_coord.standard_name == 'time'


def test_only_latitude(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix_allvar('Amon', 'mon', 'tas')

    coord_info = CoordinateInfo('latitude')
    coord_info.standard_name = 'latitude'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'latitude': coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check latitude metadata
    assert cube.coords('latitude', dim_coords=True)
    new_lat_coord = cube.coord('latitude')
    assert new_lat_coord.var_name == 'lat'
    assert new_lat_coord.standard_name == 'latitude'
    assert new_lat_coord.units == 'degrees_north'


def test_only_longitude(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix('Amon', 'mon', 'tas')

    coord_info = CoordinateInfo('longitude')
    coord_info.standard_name = 'longitude'
    monkeypatch.setattr(fix.vardef, 'coordinates', {'longitude': coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_tas_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check longitude metadata
    assert cube.coords('longitude', dim_coords=True)
    new_lon_coord = cube.coord('longitude')
    assert new_lon_coord.var_name == 'lon'
    assert new_lon_coord.standard_name == 'longitude'
    assert new_lon_coord.units == 'degrees_east'


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


def test_get_pr_fix():
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
    check_heightxm(fixed_cube, 2)

    assert fixed_cube.shape == (1, 145, 192)
