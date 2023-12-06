"""Unit tests for shared variable derivation functions."""

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor._derive._shared import (
    _get_pressure_level_widths,
    column_average,
)


def get_cube(data, air_pressure_coord=True, depth_coord=False, **kwargs):
    """Get sample cube."""
    time_coord = iris.coords.DimCoord([0.0], standard_name='time',
                                      var_name='time',
                                      units='days since 1950-01-01 00:00:00')
    plev_coord = iris.coords.DimCoord([90000.0, 80000.0],
                                      standard_name='air_pressure',
                                      var_name='plev', units='Pa')
    dpth_coord = iris.coords.DimCoord([100.0, 600.0, 7000.0],
                                      standard_name='depth',
                                      var_name='lev', units='m')
    lat_coord = iris.coords.DimCoord([45.0], standard_name='latitude',
                                     var_name='lat', units='degrees')
    lon_coord = iris.coords.DimCoord([10.0], standard_name='longitude',
                                     var_name='lon', units='degrees')
    if air_pressure_coord:
        coord_specs = [(time_coord, 0), (plev_coord, 1), (lat_coord, 2),
                       (lon_coord, 3)]
    elif depth_coord:
        coord_specs = [(time_coord, 0), (dpth_coord, 1), (lat_coord, 2),
                       (lon_coord, 3)]
    else:
        coord_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube(data, dim_coords_and_dims=coord_specs, **kwargs)
    return cube


def test_column_average():
    """Test calculation of column-average."""
    cube = get_cube([[[[1.0]], [[2.0]]]], air_pressure_coord=True,
                    var_name='ch4', units='1')
    hus_cube = get_cube([[[[0.2]], [[0.2]]]], air_pressure_coord=True,
                        var_name='hus', units='1')
    zg_cube = get_cube([[[100.0]]], air_pressure_coord=False, var_name='zg',
                       units='m')
    ps_cube = get_cube([[[100000.0]]], air_pressure_coord=False, var_name='ps',
                       units='Pa')
    x_cube = column_average(cube, hus_cube, zg_cube, ps_cube)
    assert x_cube.shape == (1, 1, 1)
    assert x_cube.units == '1'
    assert x_cube.coords('time')
    assert x_cube.coords('air_pressure')
    assert x_cube.coords('latitude')
    assert x_cube.coords('longitude')
    np.testing.assert_allclose(x_cube.data, [[[1.85]]])
    np.testing.assert_allclose(x_cube.coord('time').points, [0.0])
    np.testing.assert_allclose(x_cube.coord('air_pressure').points, 85000.0)
    np.testing.assert_allclose(x_cube.coord('air_pressure').bounds,
                               [[80000.0, 90000.0]])
    np.testing.assert_allclose(x_cube.coord('latitude').points, [45.0])
    np.testing.assert_allclose(x_cube.coord('longitude').points, [10.0])


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
