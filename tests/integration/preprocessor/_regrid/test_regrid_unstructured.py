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
def unstructured_grid_cube_2d():
    """Sample 2D cube with unstructured grid."""
    time = DimCoord(
        [0.0, 1.0], standard_name='time', units='days since 1950-01-01'
    )
    lat = AuxCoord(
        [-50.0, -50.0, 20.0, 20.0],
        standard_name='latitude',
        units='degrees_north',
    )
    lon = AuxCoord(
        [71.0, 250.0, 250.0, 71.0],
        standard_name='longitude',
        units='degrees_east',
    )
    acoord_0 = AuxCoord([0, 0], var_name='aux0')
    acoord_1 = AuxCoord([0, 0, 0, 0], var_name='aux1')
    cube = Cube(
        np.array(
            [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        ),
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
def unstructured_grid_cube_3d():
    """Sample 3D cube with unstructured grid."""
    time = DimCoord(
        [0.0, 1.0], standard_name='time', units='days since 1950-01-01'
    )
    alt = DimCoord([0.0, 1.0], standard_name='altitude', units='m')
    lat = AuxCoord(
        [-50.0, -50.0, 20.0, 20.0],
        standard_name='latitude',
        units='degrees_north',
    )
    lon = AuxCoord(
        [71.0, 250.0, 250.0, 71.0],
        standard_name='longitude',
        units='degrees_east',
    )
    acoord = AuxCoord([0, 0], var_name='aux')
    cube = Cube(
        np.ma.masked_greater(
            np.arange(16, dtype=np.float32).reshape(2, 2, 4), 7.5
        ),
        standard_name='air_temperature',
        var_name='ta',
        long_name='Air Temperature',
        units='K',
        dim_coords_and_dims=[(time, 0), (alt, 1)],
        aux_coords_and_dims=[(acoord, 1), (lat, 2), (lon, 2)],
    )
    return cube


@pytest.fixture
def target_grid():
    """Sample cube with regular grid."""
    return _global_stock_cube('120x60')


class TestUnstructuredNearest:
    """Test ``UnstructuredNearest``."""

    def test_regridding(self, unstructured_grid_cube_2d, target_grid):
        """Test regridding."""
        src_cube = unstructured_grid_cube_2d.copy()

        result = src_cube.regrid(target_grid, UnstructuredNearest())

        assert src_cube == unstructured_grid_cube_2d
        assert result.shape == (2, 3, 3)
        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        assert result.dtype == np.float32
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
        unstructured_grid_cube_2d,
        target_grid,
    ):
        """Test regridding."""
        src_cube = unstructured_grid_cube_2d.copy()
        dim_coord = DimCoord(
            [0, 1, 2, 3],
            var_name='x',
            standard_name='grid_latitude',
        )
        src_cube.add_dim_coord(dim_coord, 1)
        assert src_cube != unstructured_grid_cube_2d

        result = src_cube.regrid(target_grid, UnstructuredNearest())

        assert src_cube == unstructured_grid_cube_2d
        assert not src_cube.coords('grid_latitude')
        assert result.shape == (2, 3, 3)
        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        assert result.dtype == np.float32
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

    @pytest.mark.parametrize('units', [None, 'rad'])
    @pytest.mark.parametrize('lazy', [True, False])
    def test_regridding(
        self, lazy, units, unstructured_grid_cube_2d, target_grid,
    ):
        """Test regridding."""
        if lazy:
            unstructured_grid_cube_2d.data = (
                unstructured_grid_cube_2d.lazy_data()
            )
        if units:
            unstructured_grid_cube_2d.coord('latitude').convert_units(units)
            unstructured_grid_cube_2d.coord('longitude').convert_units(units)
            target_grid.coord('latitude').convert_units(units)
            target_grid.coord('longitude').convert_units(units)
        src_cube = unstructured_grid_cube_2d.copy()

        result = src_cube.regrid(target_grid, UnstructuredLinear())

        assert src_cube == unstructured_grid_cube_2d
        assert result.metadata == src_cube.metadata

        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        assert result.coord('aux0') == src_cube.coord('aux0')
        assert not result.coords('aux1')

        assert result.shape == (2, 3, 3)
        assert result.has_lazy_data() is lazy
        assert result.dtype == np.float32
        print(result.data)
        expected_data = np.ma.masked_invalid(
            [[
                [np.nan, np.nan, np.nan],
                [2.0820837020874023, 2.105347156524658, 1.4380426406860352],
                [np.nan, np.nan, np.nan],
            ], [
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0],
                [np.nan, np.nan, np.nan],
            ]]
        )
        np.testing.assert_allclose(result.data, expected_data)
        np.testing.assert_array_equal(result.data.mask, expected_data.mask)

    @pytest.mark.parametrize('units', [None, 'rad'])
    @pytest.mark.parametrize('lazy', [True, False])
    def test_regridding_mask_and_transposed(
        self, units, lazy, unstructured_grid_cube_3d, target_grid
    ):
        """Test regridding."""
        # Test that regridding also works if lat/lon are not rightmost
        # dimensions
        unstructured_grid_cube_3d.transpose([0, 2, 1])
        if lazy:
            unstructured_grid_cube_3d.data = (
                unstructured_grid_cube_3d.lazy_data()
            )
        if units:
            unstructured_grid_cube_3d.coord('latitude').convert_units(units)
            unstructured_grid_cube_3d.coord('longitude').convert_units(units)
            target_grid.coord('latitude').convert_units(units)
            target_grid.coord('longitude').convert_units(units)
        src_cube = unstructured_grid_cube_3d.copy()

        result = src_cube.regrid(target_grid, UnstructuredLinear())

        assert src_cube == unstructured_grid_cube_3d
        assert result.metadata == src_cube.metadata

        assert result.coord('time') == src_cube.coord('time')
        assert result.coord('altitude') == src_cube.coord('altitude')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')
        assert result.coord('aux') == src_cube.coord('aux')

        assert result.shape == (2, 3, 3, 2)
        assert result.has_lazy_data() is lazy
        assert result.dtype == np.float32

        expected_data = np.ma.masked_all((2, 3, 3, 2), dtype=np.float32)
        expected_data[0, 1, :, :] = [
            [2.0820837020874023, 6.082083702087402],
            [2.105347156524658, 6.105347156524658],
            [1.4380426406860352, 5.438042640686035],
        ]
        print(result.data)
        np.testing.assert_allclose(result.data, expected_data)
        np.testing.assert_array_equal(result.data.mask, expected_data.mask)

    def test_scheme_repr(self):
        """Test regridding."""
        assert UnstructuredLinear().__repr__() == "UnstructuredLinear()"

    def test_invalid_src_cube(self, target_grid):
        """Test regridding."""
        msg = "Source cube .* does not have unstructured grid"
        with pytest.raises(ValueError, match=msg):
            target_grid.regrid(target_grid, UnstructuredLinear())

    def test_invalid_tgt_cube(self, unstructured_grid_cube_2d):
        """Test regridding."""
        src_cube = unstructured_grid_cube_2d
        msg = "Target cube .* does not have regular grid"
        with pytest.raises(ValueError, match=msg):
            src_cube.regrid(src_cube, UnstructuredLinear())

    @pytest.mark.parametrize('units', [None, 'rad'])
    def test_regridder_same_grid(
        self,
        units,
        unstructured_grid_cube_2d,
        unstructured_grid_cube_3d,
        target_grid,
    ):
        """Test regridding."""
        if units:
            unstructured_grid_cube_2d.coord('latitude').convert_units(units)
            unstructured_grid_cube_2d.coord('longitude').convert_units(units)
            unstructured_grid_cube_3d.coord('latitude').convert_units(units)
            unstructured_grid_cube_3d.coord('longitude').convert_units(units)
            target_grid.coord('latitude').convert_units(units)
            target_grid.coord('longitude').convert_units(units)
        cube = unstructured_grid_cube_3d.copy()
        regridder = UnstructuredLinear().regridder(
            unstructured_grid_cube_2d, target_grid
        )
        result = regridder(cube)
        assert result.shape == (2, 2, 3, 3)
        assert result.coord('time') == cube.coord('time')
        assert result.coord('altitude') == cube.coord('altitude')
        assert result.coord('latitude') == target_grid.coord('latitude')
        assert result.coord('longitude') == target_grid.coord('longitude')

    def test_regridder_different_grid(
        self, unstructured_grid_cube_2d, unstructured_grid_cube_3d, target_grid
    ):
        """Test regridding."""
        cube = unstructured_grid_cube_3d.copy()
        cube.coord('latitude').points = [0.0, 0.0, 0.0, 0.0]
        regridder = UnstructuredLinear().regridder(
            unstructured_grid_cube_2d, target_grid
        )
        msg = (
            "The given cube .* is not defined on the same source grid as this "
            "regridder"
        )
        with pytest.raises(ValueError, match=msg):
            regridder(cube)

    def test_regridder_invalid_grid(
        self, unstructured_grid_cube_2d, target_grid
    ):
        """Test regridding."""
        regridder = UnstructuredLinear().regridder(
            unstructured_grid_cube_2d, target_grid
        )
        msg = "Cube .* does not have unstructured grid"
        with pytest.raises(ValueError, match=msg):
            regridder(target_grid)
