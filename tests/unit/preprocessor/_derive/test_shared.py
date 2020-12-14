"""Unit tests for shared variable derivation functions."""

import numpy as np
import pytest

from esmvalcore.preprocessor._derive._shared import _get_pressure_level_widths


def test_col_is_not_monotonic():
    """Test for non-monotonic column."""
    plev = 1000
    top_limit = 5
    col = np.array([1, 2, 3, 2, 1])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(col, air_pressure_axis=0)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(np.atleast_2d(col), air_pressure_axis=1)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(np.atleast_3d(col), air_pressure_axis=1)


def test_keeping_column_length():
    """Test for level widths keeping column length."""
    plev = 1000
    top_limit = 5
    col = np.array([1000, 900, 800])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    assert (len(_get_pressure_level_widths(col, air_pressure_axis=0)) ==
            len(col) - 2)
    col = np.atleast_2d(col)
    assert (_get_pressure_level_widths(col, air_pressure_axis=1).shape ==
            (1, 3))
    col = np.atleast_3d(col)
    assert (_get_pressure_level_widths(col, air_pressure_axis=1).shape ==
            (1, 3, 1))


def test_low_lev_surf_press():
    """Test for lowest level equal to surface pressure."""
    plev = 1000
    top_limit = 5
    col = np.array([1000, 900, 800])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    result = np.array([50, 100, 845])
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=0),
                          result)
    col = np.atleast_2d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_2d(result))
    col = np.atleast_3d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_3d(result))


def test_low_lev_above_surf_press():
    """Test for lowest level above surface pressure."""
    plev = 1020
    top_limit = 5
    col = np.array([1000, 900, 800])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    result = np.array([70, 100, 845])
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=0),
                          result)
    col = np.atleast_2d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_2d(result))
    col = np.atleast_3d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_3d(result))


def test_low_lev_below_surf_press():
    """Test for lowest level below surface pressure."""
    plev = 970
    top_limit = 5
    col = np.array([np.NaN, 900, 800])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    result = np.array([0, 120, 845])
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=0),
                          result)
    col = np.atleast_2d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_2d(result))
    col = np.atleast_3d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_3d(result))

    col = np.array([np.NaN, np.NaN, 900, 800])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    result = np.array([0, 0, 120, 845])
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=0),
                          result)
    col = np.atleast_2d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_2d(result))
    col = np.atleast_3d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_3d(result))


def test_high_level_top_limit():
    """Test for highest level equal to top limit."""
    plev = 1020
    top_limit = 5
    col = np.array([1000, 900, 5])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    result = np.array([70, 50 + 895 / 2, 895 / 2])
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=0),
                          result)
    col = np.atleast_2d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_2d(result))
    col = np.atleast_3d(col)
    assert np.array_equal(_get_pressure_level_widths(col, air_pressure_axis=1),
                          np.atleast_3d(result))


def test_high_level_above_top_limit():
    """Test for highest level above top limit."""
    plev = 1020
    top_limit = 5
    col = np.array([1000, 900, 3])
    col = np.insert(col, 0, plev)
    col = np.append(col, top_limit)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(col, air_pressure_axis=0)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(np.atleast_2d(col), air_pressure_axis=1)
    with pytest.raises(ValueError):
        _get_pressure_level_widths(np.atleast_3d(col), air_pressure_axis=1)
