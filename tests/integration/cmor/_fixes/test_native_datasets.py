"""Tests for base class of native dataset fixes."""
from unittest import mock

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes():
    """List of cubes with different `var_names`."""
    cubes = CubeList([
        Cube(0.0, var_name='pr'),
        Cube(0.0, var_name='tas'),
    ])
    return cubes


@pytest.fixture
def empty_cube():
    """Empty cube."""
    cube = Cube(1.0)
    return cube


@pytest.fixture
def sample_cube():
    """4D sample cube with many coordinates."""
    time_coord = DimCoord(
        0.0,
        long_name='time',
        units=Unit('day since 1950-01-01 00:00:00', calendar='gregorian'),
    )
    plev_coord = DimCoord(
        [1000.0, 900.0],
        long_name='air_pressure',
        units='hPa',
    )
    height_coord = AuxCoord(
        [2.0, 4.0],
        long_name='height',
        units='km',
    )
    coord_with_bounds = AuxCoord(
        [2.0, 4.0],
        bounds=[[0.0, 2.5], [2.5, 10.0]],
        long_name='coord with bounds',
        units='km',
    )
    lat_coord = DimCoord(
        3.141592653,
        long_name='latitude',
        units='rad',
    )
    lon_coord = DimCoord(
        3.141592653,
        long_name='longitude',
        units='rad',
    )
    cube = Cube(
        [[[[1.0]], [[2.0]]]],
        dim_coords_and_dims=[(time_coord, 0),
                             (plev_coord, 1),
                             (lat_coord, 2),
                             (lon_coord, 3)],
        aux_coords_and_dims=[(height_coord, 1),
                             (coord_with_bounds, 1)],
    )
    return cube


@pytest.fixture
def fix():
    """Native dataset fix.

    Note
    ----
    We use `tas` as a dummy variable here, but will use monkeypatching to
    customize the variable information of the fix in the tests below. This will
    allow us to test all common cases.

    """
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    extra_facets = {}
    fix = NativeDatasetFix(vardef, extra_facets=extra_facets)
    return fix


@pytest.mark.parametrize(
    'scalar_coord,coord_name,val',
    [
        ('height2m', 'height', 2.0),
        ('height10m', 'height', 10.0),
        ('lambda550nm', 'radiation_wavelength', 550.0),
        ('typesi', 'area_type', 'sea_ice'),
    ],
)
def test_fix_scalar_coords(monkeypatch, empty_cube, fix, scalar_coord,
                           coord_name, val):
    """Test ``fix_scalar_coords``."""
    monkeypatch.setattr(fix.vardef, 'dimensions', [scalar_coord])

    fix.fix_scalar_coords(empty_cube)

    coord = empty_cube.coord(coord_name)
    assert coord.standard_name == coord_name
    assert coord.shape == (1,)
    if isinstance(val, str):
        np.testing.assert_array_equal(coord.points, [val])
    else:
        np.testing.assert_allclose(coord.points, [val])


def test_fix_var_metadata_tas(empty_cube, fix):
    """Test ``fix_var_metadata`` using `tas`."""
    empty_cube.units = 'K'

    fix.fix_var_metadata(empty_cube)

    assert empty_cube.var_name == 'tas'
    assert empty_cube.standard_name == 'air_temperature'
    assert empty_cube.long_name == 'Near-Surface Air Temperature'
    assert empty_cube.units == 'K'
    assert 'positive' not in empty_cube.attributes


def test_fix_var_metadata_custom_var(monkeypatch, empty_cube, fix):
    """Test ``fix_var_metadata`` using custom variable."""
    monkeypatch.setattr(fix, 'INVALID_UNITS', {'invalid_units': 'kg'})
    monkeypatch.setattr(fix.vardef, 'positive', mock.sentinel.positive)
    monkeypatch.setattr(fix.vardef, 'standard_name', '')
    monkeypatch.setattr(fix.vardef, 'units', 'g')
    empty_cube.attributes['invalid_units'] = 'invalid_units'

    fix.fix_var_metadata(empty_cube)

    assert empty_cube.var_name == 'tas'
    assert empty_cube.standard_name is None
    assert empty_cube.long_name == 'Near-Surface Air Temperature'
    assert empty_cube.units == 'g'
    assert empty_cube.attributes['positive'] == mock.sentinel.positive
    np.testing.assert_allclose(empty_cube.data, 1000.0)


def test_fix_var_metadata_units_exponent(monkeypatch, empty_cube, fix):
    """Test ``fix_var_metadata`` with invalid units."""
    monkeypatch.setattr(fix.vardef, 'units', 'm s-2')
    empty_cube.attributes['invalid_units'] = 'km/s**2'

    fix.fix_var_metadata(empty_cube)

    assert empty_cube.units == 'm s-2'
    np.testing.assert_allclose(empty_cube.data, 1000.0)


def test_fix_var_metadata_units_fail(empty_cube, fix):
    """Test ``fix_var_metadata`` with invalid units."""
    empty_cube.attributes['invalid_units'] = 'invalid_units'

    msg = "Failed to fix invalid units 'invalid_units' for variable 'tas'"
    with pytest.raises(ValueError, match=msg):
        fix.fix_var_metadata(empty_cube)


def test_get_cube(cubes, fix):
    """Test ``get_cube``."""
    cube = fix.get_cube(cubes)
    assert cube.var_name == 'tas'


def test_get_cube_custom_var_name(cubes, fix):
    """Test ``get_cube`` with custom `var_name`."""
    cube = fix.get_cube(cubes, var_name='pr')
    assert cube.var_name == 'pr'


def test_get_cube_extra_facets(cubes, fix):
    """Test ``get_cube`` with `raw_name` in extra facets."""
    fix.extra_facets['raw_name'] = 'pr'
    cube = fix.get_cube(cubes)
    assert cube.var_name == 'pr'


def test_get_cube_fail(cubes, fix):
    """Test ``get_cube`` with invalid `var_name`."""
    msg = "Variable 'x' used to extract 'tas' is not available in input file"
    with pytest.raises(ValueError, match=msg):
        fix.get_cube(cubes, var_name='x')


@pytest.mark.parametrize(
    'coord,coord_name,func_name',
    [
        ('time', 'time', 'fix_regular_time'),
        ('latitude', 'latitude', 'fix_regular_lat'),
        ('longitude', 'longitude', 'fix_regular_lon'),
    ]
)
def test_fix_regular_coords_from_cube(monkeypatch, sample_cube, fix, coord,
                                      coord_name, func_name):
    """Test fixing of regular coords from cube."""
    monkeypatch.setattr(fix.vardef, 'dimensions', [coord])

    func = getattr(fix, func_name)
    func(sample_cube)

    coord = sample_cube.coord(coord_name)
    assert coord.standard_name == coord_name
    assert coord.var_name is not None
    assert coord.long_name is not None
    assert coord.bounds is None


@pytest.mark.parametrize(
    'coord,coord_name,func_name',
    [
        ('time', 'time', 'fix_regular_time'),
        ('latitude', 'latitude', 'fix_regular_lat'),
        ('longitude', 'longitude', 'fix_regular_lon'),
    ]
)
def test_fix_regular_coords_from_str(monkeypatch, sample_cube, fix, coord,
                                     coord_name, func_name):
    """Test fixing of regular coords from string."""
    monkeypatch.setattr(fix.vardef, 'dimensions', [coord])

    func = getattr(fix, func_name)
    func(sample_cube, coord=coord_name)

    coord = sample_cube.coord(coord_name)
    assert coord.standard_name == coord_name
    assert coord.var_name is not None
    assert coord.long_name is not None
    assert coord.bounds is None


@pytest.mark.parametrize(
    'func_name,coord_name,units',
    [
        ('fix_regular_time', 'time', 'days since 01-01-1990'),
        ('fix_regular_lat', 'latitude', 'rad'),
        ('fix_regular_lon', 'longitude', 'rad'),
    ]
)
def test_fix_regular_coords_from_coords(empty_cube, fix, func_name,
                                        coord_name, units):
    """Test fixing of regular coords from coords."""
    coord = AuxCoord([1.570796, 3.141592], units=units)

    func = getattr(fix, func_name)
    func(sample_cube, coord=coord)

    assert coord.standard_name == coord_name
    assert coord.var_name is not None
    assert coord.long_name is not None
    assert coord.bounds is not None


@pytest.mark.parametrize(
    'func_name,coord_name,units',
    [
        ('fix_regular_time', 'time', 'days since 01-01-1990'),
        ('fix_regular_lat', 'latitude', 'rad'),
        ('fix_regular_lon', 'longitude', 'rad'),
    ]
)
def test_fix_regular_coords_from_coords_no_bounds(empty_cube, fix, func_name,
                                                  coord_name, units):
    """Test fixing of regular coords from coords."""
    coord = AuxCoord([1.570796, 3.141592], units=units)

    func = getattr(fix, func_name)
    func(sample_cube, coord=coord, guess_bounds=False)

    assert coord.standard_name == coord_name
    assert coord.var_name is not None
    assert coord.long_name is not None
    assert coord.bounds is None


def test_guess_coord_bounds_from_str(sample_cube, fix):
    """Test ``guess_coord_bounds`` from string."""
    out_coord = fix.guess_coord_bounds(sample_cube, 'height')
    assert out_coord is sample_cube.coord('height')
    np.testing.assert_allclose(out_coord.bounds, [[1.0, 3.0], [3.0, 5.0]])


def test_guess_coord_bounds_from_str_len_1(sample_cube, fix):
    """Test ``guess_coord_bounds`` from string."""
    out_coord = fix.guess_coord_bounds(sample_cube, 'time')
    assert out_coord is sample_cube.coord('time')
    assert out_coord.bounds is None


def test_guess_coord_bounds_from_str_already_present(sample_cube, fix):
    """Test ``guess_coord_bounds`` if bounds are already present."""
    out_coord = fix.guess_coord_bounds(sample_cube, 'coord with bounds')
    assert out_coord is sample_cube.coord('coord with bounds')
    np.testing.assert_allclose(out_coord.bounds, [[0.0, 2.5], [2.5, 10.0]])


def test_guess_coord_bounds_from_coord(empty_cube, fix):
    """Test ``guess_coord_bounds`` from coord."""
    coord = AuxCoord([2.0, 4.0])
    out_coord = fix.guess_coord_bounds(empty_cube, coord)
    assert out_coord is coord
    np.testing.assert_allclose(out_coord.bounds, [[1.0, 3.0], [3.0, 5.0]])


def test_guess_coord_bounds_from_coord_len_1(empty_cube, fix):
    """Test ``guess_coord_bounds`` from coord."""
    coord = AuxCoord([2.0])
    out_coord = fix.guess_coord_bounds(empty_cube, coord)
    assert out_coord is coord
    assert out_coord.bounds is None


def test_guess_coord_bounds_from_coord_already_present(empty_cube, fix):
    """Test ``guess_coord_bounds`` if bounds are already present."""
    coord = AuxCoord([2.0, 4.0], bounds=[[0.0, 2.5], [2.5, 10.0]])
    out_coord = fix.guess_coord_bounds(empty_cube, coord)
    assert out_coord is coord
    np.testing.assert_allclose(out_coord.bounds, [[0.0, 2.5], [2.5, 10.0]])


def test_fix_time_metadata(sample_cube, fix):
    """Test ``fix_time_metadata``."""
    out_coord = fix.fix_time_metadata(sample_cube)
    assert out_coord is sample_cube.coord('time')
    assert out_coord.standard_name == 'time'
    assert out_coord.var_name == 'time'
    assert out_coord.long_name == 'time'
    assert out_coord.units == 'day since 1950-01-01 00:00:00'
    np.testing.assert_allclose(out_coord.points, [0.0])
    assert out_coord.bounds is None


def test_fix_time_metadata_from_str(sample_cube, fix):
    """Test ``fix_time_metadata`` from string."""
    out_coord = fix.fix_time_metadata(sample_cube, coord='time')
    assert out_coord is sample_cube.coord('time')
    assert out_coord.standard_name == 'time'
    assert out_coord.var_name == 'time'
    assert out_coord.long_name == 'time'
    assert out_coord.units == 'day since 1950-01-01 00:00:00'
    np.testing.assert_allclose(out_coord.points, [0.0])
    assert out_coord.bounds is None


def test_fix_time_metadata_from_coord(sample_cube, fix):
    """Test ``fix_time_metadata`` from string."""
    coord = AuxCoord([2.0], units='day since 1950-01-01 00:00:00')
    out_coord = fix.fix_time_metadata(sample_cube, coord=coord)
    assert out_coord is coord
    assert out_coord.standard_name == 'time'
    assert out_coord.var_name == 'time'
    assert out_coord.long_name == 'time'
    assert out_coord.units == 'day since 1950-01-01 00:00:00'
    np.testing.assert_allclose(out_coord.points, [2.0])
    assert out_coord.bounds is None


def test_fix_height_metadata(sample_cube, fix):
    """Test ``fix_height_metadata``."""
    out_coord = fix.fix_height_metadata(sample_cube)
    assert out_coord is sample_cube.coord('height')
    assert out_coord.standard_name == 'height'
    assert out_coord.var_name == 'height'
    assert out_coord.long_name == 'height'
    assert out_coord.units == 'm'
    assert out_coord.attributes['positive'] == 'up'
    np.testing.assert_allclose(out_coord.points, [2000.0, 4000.0])
    assert out_coord.bounds is None


def test_fix_height_metadata_from_str(sample_cube, fix):
    """Test ``fix_height_metadata`` from string."""
    out_coord = fix.fix_height_metadata(sample_cube, coord='height')
    assert out_coord is sample_cube.coord('height')
    assert out_coord.standard_name == 'height'
    assert out_coord.var_name == 'height'
    assert out_coord.long_name == 'height'
    assert out_coord.units == 'm'
    assert out_coord.attributes['positive'] == 'up'
    np.testing.assert_allclose(out_coord.points, [2000.0, 4000.0])
    assert out_coord.bounds is None


def test_fix_height_metadata_from_coord(sample_cube, fix):
    """Test ``fix_height_metadata`` from string."""
    coord = AuxCoord([2.0], units='m')
    out_coord = fix.fix_height_metadata(sample_cube, coord=coord)
    assert out_coord is coord
    assert out_coord.standard_name == 'height'
    assert out_coord.var_name == 'height'
    assert out_coord.long_name == 'height'
    assert out_coord.units == 'm'
    assert out_coord.attributes['positive'] == 'up'
    np.testing.assert_allclose(out_coord.points, [2.0])
    assert out_coord.bounds is None


def test_fix_plev_metadata(sample_cube, fix):
    """Test ``fix_plev_metadata``."""
    out_coord = fix.fix_plev_metadata(sample_cube)
    assert out_coord is sample_cube.coord('air_pressure')
    assert out_coord.standard_name == 'air_pressure'
    assert out_coord.var_name == 'plev'
    assert out_coord.long_name == 'pressure'
    assert out_coord.units == 'Pa'
    assert out_coord.attributes['positive'] == 'down'
    np.testing.assert_allclose(out_coord.points, [100000.0, 90000.0])
    assert out_coord.bounds is None


def test_fix_plev_metadata_from_str(sample_cube, fix):
    """Test ``fix_plev_metadata`` from string."""
    out_coord = fix.fix_plev_metadata(sample_cube, coord='air_pressure')
    assert out_coord is sample_cube.coord('air_pressure')
    assert out_coord.standard_name == 'air_pressure'
    assert out_coord.var_name == 'plev'
    assert out_coord.long_name == 'pressure'
    assert out_coord.units == 'Pa'
    assert out_coord.attributes['positive'] == 'down'
    np.testing.assert_allclose(out_coord.points, [100000.0, 90000.0])
    assert out_coord.bounds is None


def test_fix_plev_metadata_from_coord(sample_cube, fix):
    """Test ``fix_plev_metadata`` from string."""
    coord = AuxCoord([1.0], units='Pa')
    out_coord = fix.fix_plev_metadata(sample_cube, coord=coord)
    assert out_coord is coord
    assert out_coord.standard_name == 'air_pressure'
    assert out_coord.var_name == 'plev'
    assert out_coord.long_name == 'pressure'
    assert out_coord.units == 'Pa'
    assert out_coord.attributes['positive'] == 'down'
    np.testing.assert_allclose(out_coord.points, [1.0])
    assert out_coord.bounds is None


def test_fix_lat_metadata(sample_cube, fix):
    """Test ``fix_lat_metadata``."""
    out_coord = fix.fix_lat_metadata(sample_cube)
    assert out_coord is sample_cube.coord('latitude')
    assert out_coord.standard_name == 'latitude'
    assert out_coord.var_name == 'lat'
    assert out_coord.long_name == 'latitude'
    assert out_coord.units == 'degrees_north'
    np.testing.assert_allclose(out_coord.points, [180.0])
    assert out_coord.bounds is None


def test_fix_lat_metadata_from_str(sample_cube, fix):
    """Test ``fix_lat_metadata`` from string."""
    out_coord = fix.fix_lat_metadata(sample_cube, coord='latitude')
    assert out_coord is sample_cube.coord('latitude')
    assert out_coord.standard_name == 'latitude'
    assert out_coord.var_name == 'lat'
    assert out_coord.long_name == 'latitude'
    assert out_coord.units == 'degrees_north'
    np.testing.assert_allclose(out_coord.points, [180.0])
    assert out_coord.bounds is None


def test_fix_lat_metadata_from_coord(sample_cube, fix):
    """Test ``fix_lat_metadata`` from string."""
    coord = AuxCoord([0.0], units='degrees')
    out_coord = fix.fix_lat_metadata(sample_cube, coord=coord)
    assert out_coord is coord
    assert out_coord.standard_name == 'latitude'
    assert out_coord.var_name == 'lat'
    assert out_coord.long_name == 'latitude'
    assert out_coord.units == 'degrees_north'
    np.testing.assert_allclose(out_coord.points, [0.0])
    assert out_coord.bounds is None


def test_fix_lon_metadata(sample_cube, fix):
    """Test ``fix_lon_metadata``."""
    out_coord = fix.fix_lon_metadata(sample_cube)
    assert out_coord is sample_cube.coord('longitude')
    assert out_coord.standard_name == 'longitude'
    assert out_coord.var_name == 'lon'
    assert out_coord.long_name == 'longitude'
    assert out_coord.units == 'degrees_east'
    np.testing.assert_allclose(out_coord.points, [180.0])
    assert out_coord.bounds is None


def test_fix_lon_metadata_from_str(sample_cube, fix):
    """Test ``fix_lon_metadata`` from string."""
    out_coord = fix.fix_lon_metadata(sample_cube, coord='longitude')
    assert out_coord is sample_cube.coord('longitude')
    assert out_coord.standard_name == 'longitude'
    assert out_coord.var_name == 'lon'
    assert out_coord.long_name == 'longitude'
    assert out_coord.units == 'degrees_east'
    np.testing.assert_allclose(out_coord.points, [180.0])
    assert out_coord.bounds is None


def test_fix_lon_metadata_from_coord(sample_cube, fix):
    """Test ``fix_lon_metadata`` from string."""
    coord = AuxCoord([0.0], units='degrees')
    out_coord = fix.fix_lon_metadata(sample_cube, coord=coord)
    assert out_coord is coord
    assert out_coord.standard_name == 'longitude'
    assert out_coord.var_name == 'lon'
    assert out_coord.long_name == 'longitude'
    assert out_coord.units == 'degrees_east'
    np.testing.assert_allclose(out_coord.points, [0.0])
    assert out_coord.bounds is None
