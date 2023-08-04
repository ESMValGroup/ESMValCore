"""Unit tests for automatic fixes."""

from unittest.mock import sentinel

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.automatic_fix import AutomaticFix, get_time_bounds


@pytest.fixture
def time_coord():
    """Time coordinate."""
    time_coord = AuxCoord(
        [15, 350],
        standard_name='time',
        units='days since 1850-01-01'
    )
    return time_coord


@pytest.fixture
def automatic_fix():
    """Automatic fix object."""
    return AutomaticFix.from_facets('CMIP6', 'CFmon', 'ta')


@pytest.mark.parametrize(
    'freq,expected_bounds',
    [
        ('mon', [[0, 31], [334, 365]]),
        ('mo', [[0, 31], [334, 365]]),
        ('yr', [[0, 365], [0, 365]]),
        ('dec', [[0, 3652], [0, 3652]]),
        ('day', [[14.5, 15.5], [349.5, 350.5]]),
        ('6hr', [[14.875, 15.125], [349.875, 350.125]]),
        ('3hr', [[14.9375, 15.0625], [349.9375, 350.0625]]),
        ('1hr', [[14.97916666, 15.020833333], [349.97916666, 350.020833333]]),
    ]
)
def test_get_time_bounds(time_coord, freq, expected_bounds):
    """Test ``get_time_bounds`."""
    bounds = get_time_bounds(time_coord, freq)
    np.testing.assert_allclose(bounds, expected_bounds)


def test_get_time_bounds_invalid_freq_fail(time_coord):
    """Test ``get_time_bounds`."""
    with pytest.raises(NotImplementedError):
        get_time_bounds(time_coord, 'invalid_freq')


def test_automatic_fix_var_info_none():
    """Test ``AutomaticFix``."""
    with pytest.raises(ValueError):
        AutomaticFix(None)


def test_automatic_fix_empty_long_name(automatic_fix, monkeypatch):
    """Test ``AutomaticFix``."""
    # Artificially set long_name to empty string for test
    monkeypatch.setattr(automatic_fix.var_info, 'long_name', '')

    cube = automatic_fix.fix_long_name(sentinel.cube)

    assert cube == sentinel.cube


def test_automatic_fix_empty_units(automatic_fix, monkeypatch):
    """Test ``AutomaticFix``."""
    # Artificially set latitude units to empty string for test
    coord_info = automatic_fix.var_info.coordinates['latitude']
    monkeypatch.setattr(coord_info, 'units', '')

    ret = automatic_fix.fix_coord_units(
        sentinel.cube, coord_info, sentinel.cube_coord
    )

    assert ret is None


def test_automatic_fix_no_generic_lev_coords(automatic_fix, monkeypatch):
    """Test ``AutomaticFix``."""
    # Artificially remove generic_lev_coords
    monkeypatch.setattr(
        automatic_fix.var_info.coordinates['alevel'], 'generic_lev_coords', {}
    )

    cube = automatic_fix.fix_alternative_generic_level_coords(sentinel.cube)

    assert cube == sentinel.cube


def test_requested_levels_2d_coord(automatic_fix, mocker):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord([[0]], standard_name='latitude', units='rad')
    cmor_coord = mocker.Mock(requested=True)

    ret = automatic_fix.fix_requested_coord_values(
        sentinel.cube, cmor_coord, cube_coord
    )

    assert ret is None


def test_requested_levels_invalid_arr(automatic_fix, mocker):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord([0], standard_name='latitude', units='rad')
    cmor_coord = mocker.Mock(requested=['a', 'b'])

    ret = automatic_fix.fix_requested_coord_values(
        sentinel.cube, cmor_coord, cube_coord
    )

    assert ret is None


def test_lon_no_fix_needed(automatic_fix):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord(
        [0.0, 180.0, 360.0], standard_name='longitude', units='rad'
    )

    ret = automatic_fix.fix_longitude_0_360(
        sentinel.cube, sentinel.cmor_coord, cube_coord
    )

    assert ret == (sentinel.cube, cube_coord)


def test_lon_too_low_to_fix(automatic_fix):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord(
        [-370.0, 0.0], standard_name='longitude', units='rad'
    )

    ret = automatic_fix.fix_longitude_0_360(
        sentinel.cube, sentinel.cmor_coord, cube_coord
    )

    assert ret == (sentinel.cube, cube_coord)


def test_lon_too_high_to_fix(automatic_fix):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord([750.0, 0.0], standard_name='longitude', units='rad')

    ret = automatic_fix.fix_longitude_0_360(
        sentinel.cube, sentinel.cmor_coord, cube_coord
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_2d_coord(automatic_fix):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord([[0]], standard_name='latitude', units='rad')

    ret = automatic_fix.fix_coord_direction(
        sentinel.cube, sentinel.cmor_coord, cube_coord
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_string_coord(automatic_fix):
    """Test ``AutomaticFix``."""
    cube_coord = AuxCoord(['a'], standard_name='latitude', units='rad')

    ret = automatic_fix.fix_coord_direction(
        sentinel.cube, sentinel.cmor_coord, cube_coord
    )

    assert ret == (sentinel.cube, cube_coord)


def test_fix_direction_no_stored_direction(automatic_fix, mocker):
    """Test ``AutomaticFix``."""
    cube = Cube(0)
    cube_coord = AuxCoord([0, 1], standard_name='latitude', units='rad')
    cmor_coord = mocker.Mock(stored_direction='')

    ret = automatic_fix.fix_coord_direction(cube, cmor_coord, cube_coord)

    assert ret == (cube, cube_coord)
