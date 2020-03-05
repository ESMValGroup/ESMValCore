"""Unit tests for regridding multidimensional :class:`iris.coords.AuxCoord`."""

import unittest
from unittest.mock import call, sentinel

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from esmvalcore.preprocessor._regrid import _add_regridded_aux_coords


def mock_coord(name, points, bounds=None):
    """Return mocked coordinate."""
    coord = unittest.mock.create_autospec(AuxCoord, spec_set=True,
                                          instance=True)
    points = np.array(points)
    if bounds is not None:
        bounds = np.array(bounds)
    coord.ndim = points.ndim
    coord.shape = points.shape
    coord.points = points
    coord.bounds = bounds
    coord.name.return_value = name
    coord.core_points.return_value = points
    coord.core_bounds.return_value = bounds
    coord.copy.return_value = sentinel.copied_aux_coord
    return coord


@pytest.fixture
def mock_1d_cube_empty():
    """Return mocked 1D cube with no coordinates."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    cube.ndim = 1
    cube.shape = (1,)
    cube.coords.return_value = []
    cube.coord_dims.return_value = ()
    cube.copy.return_value = cube
    return cube


@pytest.fixture
def mock_2d_cube_empty():
    """Return mocked 2D cube with no coordinates."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    cube.ndim = 2
    cube.shape = (1, 1)
    cube.coords.return_value = []
    cube.coord_dims.return_value = ()
    cube.copy.return_value = cube
    return cube


@pytest.fixture
def mock_1d_cube():
    """Return mocked 1D cube."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    cube.ndim = 1
    cube.shape = (1,)
    no_bounds_coord = mock_coord('no_bnds', [0.0])
    bounds_coord = mock_coord('bnds', [0.0], bounds=[[-1.0, 1.0]])
    cube.coords.return_value = [no_bounds_coord, bounds_coord]
    cube.coord_dims.return_value = (0,)
    cube.copy.return_value = cube
    cube.regrid.return_value = cube
    return cube


@pytest.fixture
def mock_2d_cube_no_bounds():
    """Return mocked 2D cube without bounds."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    cube.ndim = 2
    cube.shape = (1, 1)
    a_coord = mock_coord('a', [[0.0]])
    cube.coords.return_value = [a_coord]
    cube.coord_dims.return_value = (0, 1)
    cube.copy.return_value = cube
    cube.regrid.return_value = cube
    return cube


@pytest.fixture
def mock_2d_cube_bounds():
    """Return mocked 2D cube with bounds."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    cube.ndim = 2
    cube.shape = (1, 1)
    a_coord = mock_coord('a', [[0.0]], bounds=[[[-0.5, 0.5]]])
    cube.coords.return_value = [a_coord]
    cube.coord_dims.return_value = (0, 1)
    cube.copy.return_value = cube
    cube.regrid.return_value = cube
    return cube


@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_unequal_ndims(mock_logger,
                                                mock_1d_cube_empty,
                                                mock_2d_cube_no_bounds):
    """Test regriddind of aux coords when cubes' ndims differ."""
    _add_regridded_aux_coords(mock_1d_cube_empty, mock_2d_cube_no_bounds,
                              sentinel.target_grid)
    assert mock_1d_cube_empty.mock_calls == []
    assert mock_2d_cube_no_bounds.mock_calls == []
    mock_logger.warning.assert_not_called()


@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_empty(mock_logger, mock_2d_cube_empty):
    """Test regriddind of aux coords with empty cube."""
    _add_regridded_aux_coords(mock_2d_cube_empty, mock_2d_cube_empty,
                              sentinel.target_grid)
    assert mock_2d_cube_empty.mock_calls == [call.coords(dim_coords=False)]
    mock_logger.warning.assert_not_called()


@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_coords_present(mock_logger,
                                                 mock_2d_cube_no_bounds):
    """Test regridding of aux coords when already in regridded cube."""
    _add_regridded_aux_coords(mock_2d_cube_no_bounds, mock_2d_cube_no_bounds,
                              sentinel.target_grid)
    expected_calls = [call.coords(dim_coords=False), call.coords('a')]
    assert mock_2d_cube_no_bounds.mock_calls == expected_calls
    mock_logger.warning.assert_not_called()


@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_2d_with_bounds(mock_logger,
                                                 mock_2d_cube_bounds,
                                                 mock_2d_cube_empty):
    """Test regridding of aux coords with 2D cube with bounds."""
    _add_regridded_aux_coords(mock_2d_cube_bounds, mock_2d_cube_empty,
                              sentinel.target_grid)
    assert mock_2d_cube_bounds.mock_calls == [call.coords(dim_coords=False)]
    assert mock_2d_cube_empty.mock_calls == [call.coords('a')]
    mock_logger.warning.assert_called_once()


@unittest.mock.patch('esmvalcore.preprocessor._regrid.Linear', autospec=True)
@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_2d_without_bounds(mock_logger,
                                                    mock_linear,
                                                    mock_2d_cube_no_bounds,
                                                    mock_2d_cube_empty):
    """Test regridding of aux coords with 2D cube without bounds."""
    mock_linear.return_value = sentinel.linear
    _add_regridded_aux_coords(mock_2d_cube_no_bounds, mock_2d_cube_empty,
                              sentinel.target_grid)
    mock_2d_cube_no_bounds.regrid.assert_called_once_with(
        sentinel.target_grid, sentinel.linear)
    mock_2d_cube_empty.add_aux_coord.assert_called_once_with(
        sentinel.copied_aux_coord, (0, 1))
    mock_logger.warning.assert_not_called()


@unittest.mock.patch('esmvalcore.preprocessor._regrid.Linear', autospec=True)
@unittest.mock.patch('esmvalcore.preprocessor._regrid.logger', autospec=True)
def test_add_regridded_aux_coords_1d(mock_logger, mock_linear, mock_1d_cube,
                                     mock_1d_cube_empty):
    """Test regridding of aux coords with 1D cube."""
    mock_linear.return_value = sentinel.linear
    _add_regridded_aux_coords(mock_1d_cube, mock_1d_cube_empty,
                              sentinel.target_grid)
    mock_1d_cube.regrid.assert_called_once_with(sentinel.target_grid,
                                                sentinel.linear)
    mock_1d_cube_empty.add_aux_coord.assert_called_once_with(
        sentinel.copied_aux_coord, (0,))
    mock_logger.warning.assert_called_once()
