""" Integration tests for `bilinear_unstructured_regrid`."""

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

import esmvalcore.preprocessor._regrid
from esmvalcore.preprocessor._regrid import (
    _bilinear_unstructured_regrid,
    _get_linear_interpolation_weights,
)


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """Start each test with a clear cache."""
    monkeypatch.setattr(esmvalcore.preprocessor._regrid, '_CACHE_WEIGHTS', {})


@pytest.fixture
def unstructured_grid_cube():
    """Sample cube with unstructured grid."""
    time = DimCoord(
        [0.0, 1.0], standard_name='time', units='days since 1950-01-01'
    )
    lat = AuxCoord(
        [-1.0, -1.0, 1.0, 1.0], standard_name='latitude', units='degrees_north'
    )
    lon = AuxCoord(
        [179.0, 180.0, 180.0, 179.0],
        standard_name='longitude',
        units='degrees_east',
    )
    cube = Cube(
        np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        standard_name='air_temperature',
        units='K',
        dim_coords_and_dims=[(time, 0)],
        aux_coords_and_dims=[(lat, 1), (lon, 1)],
    )
    return cube


TARGET_GRID = '120x60'
LAT_OFFSET = True
LON_OFFSET = True


def test_use_cached_weights(unstructured_grid_cube, mocker):
    """Test `_get_linear_interpolation_weights`."""
    cache = esmvalcore.preprocessor._regrid._CACHE_WEIGHTS
    key = (
        '(4,)_-1.0-1.0-degrees_north_179.0-179.0-degrees_east_120x60_'
        'True_True_nan_'
    )
    cache[key] = mocker.sentinel.cached_weights

    weights = _get_linear_interpolation_weights(
        unstructured_grid_cube, TARGET_GRID, LAT_OFFSET, LON_OFFSET
    )

    assert weights == mocker.sentinel.cached_weights


def test_bilinear_unstructured_regrid(unstructured_grid_cube):
    """Test `_bilinear_unstructured_regrid`."""
    new_cube = _bilinear_unstructured_regrid(
        unstructured_grid_cube, TARGET_GRID, LAT_OFFSET, LON_OFFSET
    )

    assert new_cube.metadata == unstructured_grid_cube.metadata
    assert new_cube.shape == (2, 3, 3)

    assert new_cube.coords('time')
    assert new_cube.coord('time') == unstructured_grid_cube.coord('time')

    assert new_cube.coords('latitude')
    lat = new_cube.coord('latitude')
    np.testing.assert_allclose(lat.points, [-60, 0, 60])
    np.testing.assert_allclose(lat.bounds, [[-90, -30], [-30, 30], [30, 90]])

    assert new_cube.coords('longitude')
    lat = new_cube.coord('longitude')
    np.testing.assert_allclose(lat.points, [60, 180, 300])
    np.testing.assert_allclose(lat.bounds, [[0, 120], [120, 240], [240, 360]])

    np.testing.assert_allclose(
        new_cube.data,
        [
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 1.5, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan],
                [np.nan, np.nan, np.nan],
            ],
        ],
    )

    cache = esmvalcore.preprocessor._regrid._CACHE_WEIGHTS
    assert len(cache) == 1
    key = (
        '(4,)_-1.0-1.0-degrees_north_179.0-179.0-degrees_east_120x60_'
        'True_True_nan_'
    )
    assert key in cache
    assert len(cache[key]) == 2
    np.testing.assert_equal(
        cache[key][0],
        [[3, 1, 0],
         [3, 1, 0],
         [3, 1, 0],
         [3, 1, 0],
         [1, 3, 2],
         [3, 1, 0],
         [3, 1, 0],
         [3, 1, 0],
         [3, 1, 0]],
    )
    np.testing.assert_allclose(
        cache[key][1],
        [[np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [0.5, 0.0, 0.5],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan],
         [np.nan, np.nan, np.nan]],
    )


def test_bilinear_unstructured_regrid_no_unstructured_grid():
    """Test `_bilinear_unstructured_regrid`."""
    with pytest.raises(ValueError):
        _bilinear_unstructured_regrid(
            Cube(0), TARGET_GRID, LAT_OFFSET, LON_OFFSET
        )


def test_bilinear_unstructured_regrid_invalid_dims(unstructured_grid_cube):
    """Test `_bilinear_unstructured_regrid`."""
    cube = unstructured_grid_cube.copy()
    cube.transpose()
    with pytest.raises(ValueError):
        _bilinear_unstructured_regrid(
            cube, TARGET_GRID, LAT_OFFSET, LON_OFFSET
        )
