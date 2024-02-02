""" Integration tests for unstructured regridding."""

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from esmvalcore.preprocessor._regrid import _global_stock_cube
from esmvalcore.preprocessor._regrid_unstructured import UnstructuredNearest


@pytest.fixture
def unstructured_grid_cube():
    """Sample cube with unstructured grid."""
    time = DimCoord(
        [0.0, 1.0], standard_name='time', units='days since 1950-01-01'
    )
    lat = AuxCoord(
        [-50.0, -50.0, 20.0, 20.0],
        standard_name='latitude',
        units='degrees_north',
    )
    lon = AuxCoord(
        [70.0, 250.0, 250.0, 70.0],
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


@pytest.fixture
def target_grid():
    """Sample cube with regular grid."""
    return _global_stock_cube('120x60')


class TestUnstructuredNearest:
    """Test ``UnstructuredNearest``."""

    def test_regridding(self, unstructured_grid_cube, target_grid):
        """Test regridding."""
        src_cube = unstructured_grid_cube.copy()

        result = src_cube.regrid(target_grid, UnstructuredNearest())

        assert src_cube == unstructured_grid_cube
        assert result.shape == (2, 3, 3)
        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        np.testing.assert_allclose(
            result.data,
            [[[0.0, 1.0, 1.0],
              [3.0, 2.0, 2.0],
              [3.0, 2.0, 2.0]],
             [[0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]]],
        )

    def test_regridding_with_dim_coord(
        self,
        unstructured_grid_cube,
        target_grid,
    ):
        """Test regridding."""
        src_cube = unstructured_grid_cube.copy()
        dim_coord = DimCoord(
            [0, 1, 2, 3],
            var_name='x',
            standard_name='grid_latitude',
        )
        src_cube.add_dim_coord(dim_coord, 1)
        assert src_cube != unstructured_grid_cube

        result = src_cube.regrid(target_grid, UnstructuredNearest())

        assert src_cube == unstructured_grid_cube
        assert not src_cube.coords('grid_latitude')
        assert result.shape == (2, 3, 3)
        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        np.testing.assert_allclose(
            result.data,
            [[[0.0, 1.0, 1.0],
              [3.0, 2.0, 2.0],
              [3.0, 2.0, 2.0]],
             [[0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]]],
        )
