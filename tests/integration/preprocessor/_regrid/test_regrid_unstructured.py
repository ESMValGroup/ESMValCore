""" Integration tests for unstructured regridding."""

import numpy as np
import pytest
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube

from esmvalcore.preprocessor._regrid import _global_stock_cube
from esmvalcore.preprocessor._regrid_unstructured import (
    UnstructuredLinear,
    UnstructuredNearest,
)


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
    acoord_0 = AuxCoord([0, 0], var_name='aux0')
    acoord_1 = AuxCoord([0, 0, 0, 0], var_name='aux1')
    cube = Cube(
        np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        standard_name='air_temperature',
        var_name='ta',
        long_name='Air Temperature',
        units='K',
        dim_coords_and_dims=[(time, 0)],
        aux_coords_and_dims=[(acoord_0, 0), (acoord_1, 1), (lat, 1), (lon, 1)],
        attributes={'test': '1'},
        cell_methods=(CellMethod('test', 'time'),),
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


class TestUnstructuredLinear:
    """Test ``UnstructuredLinear``."""

    @pytest.mark.parametrize('lazy', [True, False])
    def test_regridding(self, lazy, unstructured_grid_cube, target_grid):
        """Test regridding."""
        if lazy:
            unstructured_grid_cube.data = unstructured_grid_cube.lazy_data()
        src_cube = unstructured_grid_cube.copy()

        result = src_cube.regrid(target_grid, UnstructuredLinear())

        assert src_cube == unstructured_grid_cube
        assert result.metadata == src_cube.metadata

        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        assert result.coord('aux0') == src_cube.coord('aux0')
        assert not result.coords('aux1')

        assert result.shape == (2, 3, 3)
        assert result.has_lazy_data() is lazy
        expected_data = np.ma.masked_invalid(
            [[
                [np.nan, np.nan, np.nan],
                [np.nan, 2.1031746031746033, np.nan],
                [np.nan, np.nan, np.nan],
            ], [
                [np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan],
                [np.nan, np.nan, np.nan],
            ]]
        )
        np.testing.assert_allclose(result.data, expected_data)
        np.testing.assert_allclose(result.data.mask, expected_data.mask)
