"""
Integration tests for the :func:`esmvalcore.preprocessor.regrid.regrid`
function.

"""

import iris
import numpy as np
from numpy import ma

import tests
from esmvalcore.preprocessor import regrid
from esmvalcore.preprocessor._io import GLOBAL_FILL_VALUE
from tests.unit.preprocessor._regrid import _make_cube


class Test(tests.Test):
    def setUp(self):
        """Prepare tests."""
        shape = (3, 2, 2)
        data = np.arange(np.prod(shape, dtype=float)).reshape(shape)
        self.cube = _make_cube(data)
        self.cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

        # Setup grid for linear regridding
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord([1.5],
                                    standard_name='longitude',
                                    bounds=[[1, 2]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.5],
                                    standard_name='latitude',
                                    bounds=[[1, 2]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        self.grid_for_linear = grid

        # Setup unstructured cube and grid
        data = np.zeros((1, 1))
        lons = iris.coords.DimCoord([1.6],
                                    standard_name='longitude',
                                    bounds=[[1, 2]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.6],
                                    standard_name='latitude',
                                    bounds=[[1, 2]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        self.grid_for_unstructured_nearest = iris.cube.Cube(
            data, dim_coords_and_dims=coords_spec)

        # Replace 1d spatial coords with 2d spatial coords.
        lons = self.cube.coord('longitude')
        lats = self.cube.coord('latitude')
        x, y = np.meshgrid(lons.points, lats.points)

        lats = iris.coords.AuxCoord(
            y,
            standard_name=lats.metadata.standard_name,
            long_name=lats.metadata.long_name,
            var_name=lats.metadata.var_name,
            units=lats.metadata.units,
            attributes=lats.metadata.attributes,
            coord_system=lats.metadata.coord_system,
            climatological=lats.metadata.climatological)

        lons = iris.coords.AuxCoord(
            x,
            standard_name=lons.metadata.standard_name,
            long_name=lons.metadata.long_name,
            var_name=lons.metadata.var_name,
            units=lons.metadata.units,
            attributes=lons.metadata.attributes,
            coord_system=lons.metadata.coord_system,
            climatological=lons.metadata.climatological)

        self.unstructured_grid_cube = self.cube.copy()
        self.unstructured_grid_cube.remove_coord('longitude')
        self.unstructured_grid_cube.remove_coord('latitude')
        self.unstructured_grid_cube.remove_coord('Pressure Slice')
        self.unstructured_grid_cube.add_aux_coord(lons, (1, 2))
        self.unstructured_grid_cube.add_aux_coord(lats, (1, 2))
        self.unstructured_grid_cube.data = self.cube.data.astype(np.float32)

    def test_regrid__linear(self):
        result = regrid(self.cube, self.grid_for_linear, 'linear')
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        self.assert_array_equal(result.data, expected)

    def test_regrid__regular_coordinates(self):
        data = np.ones((1, 1))
        lons = iris.coords.DimCoord([1.50000000000001],
                                    standard_name='longitude',
                                    bounds=[[1, 2]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.50000000000001],
                                    standard_name='latitude',
                                    bounds=[[1, 2]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        regular_grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(regular_grid, self.grid_for_linear, 'linear')
        iris.common.resolve.Resolve(result, self.grid_for_linear)

    def test_regrid__linear_do_not_preserve_dtype(self):
        self.cube.data = self.cube.data.astype(int)
        result = regrid(self.cube, self.grid_for_linear, 'linear')
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        self.assert_array_equal(result.data, expected)
        assert np.issubdtype(self.cube.dtype, np.integer)
        assert np.issubdtype(result.dtype, np.floating)

    def test_regrid__linear_extrapolate(self):
        data = np.empty((3, 3))
        lons = iris.coords.DimCoord([0, 1.5, 3],
                                    standard_name='longitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([0, 1.5, 3],
                                    standard_name='latitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, 'linear_extrapolate')
        expected = [[[-3., -1.5, 0.], [0., 1.5, 3.], [3., 4.5, 6.]],
                    [[1., 2.5, 4.], [4., 5.5, 7.], [7., 8.5, 10.]],
                    [[5., 6.5, 8.], [8., 9.5, 11.], [11., 12.5, 14.]]]
        self.assert_array_equal(result.data, expected)

    def test_regrid__linear_extrapolate_with_mask(self):
        data = np.empty((3, 3))
        grid = iris.cube.Cube(data)
        lons = iris.coords.DimCoord([0, 1.5, 3],
                                    standard_name='longitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([0, 1.5, 3],
                                    standard_name='latitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, 'linear')
        expected = ma.empty((3, 3, 3))
        expected.mask = ma.masked
        expected[:, 1, 1] = np.array([1.5, 5.5, 9.5])
        self.assert_array_equal(result.data, expected)

    def test_regrid__nearest(self):
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord([1.6],
                                    standard_name='longitude',
                                    bounds=[[1, 2]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.6],
                                    standard_name='latitude',
                                    bounds=[[1, 2]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, 'nearest')
        expected = np.array([[[3]], [[7]], [[11]]])
        self.assert_array_equal(result.data, expected)

    def test_regrid__nearest_extrapolate_with_mask(self):
        data = np.empty((3, 3))
        lons = iris.coords.DimCoord([0, 1.6, 3],
                                    standard_name='longitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([0, 1.6, 3],
                                    standard_name='latitude',
                                    bounds=[[0, 1], [1, 2], [2, 3]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, 'nearest')
        expected = ma.empty((3, 3, 3))
        expected.mask = ma.masked
        expected[:, 1, 1] = np.array([3, 7, 11])
        self.assert_array_equal(result.data, expected)

    def test_regrid__area_weighted(self):
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord([1.6],
                                    standard_name='longitude',
                                    bounds=[[1, 2]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.6],
                                    standard_name='latitude',
                                    bounds=[[1, 2]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, 'area_weighted')
        expected = np.array([1.499886, 5.499886, 9.499886])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=6)

    def test_regrid__unstructured_nearest_float(self):
        """Test unstructured_nearest regridding with cube of floats."""
        result = regrid(self.unstructured_grid_cube,
                        self.grid_for_unstructured_nearest,
                        'unstructured_nearest')
        expected = np.array([[[3.0]], [[7.0]], [[11.0]]])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=6)

        # Make sure that dtype is preserved (without an adaption in
        # esmvalcore.preprocessor.regrid(), the dtype of the result would be
        # float64 instead of float32)
        assert self.unstructured_grid_cube.dtype == np.float32
        assert result.dtype == np.float32

        # Make sure that output is a masked array with correct fill value
        # (= GLOBAL_FILL_VALUE)
        np.testing.assert_allclose(result.data.fill_value, GLOBAL_FILL_VALUE)

    def test_regrid__unstructured_nearest_int(self):
        """Test unstructured_nearest regridding with cube of ints."""
        self.unstructured_grid_cube.data = np.full((3, 2, 2), 1, dtype=int)
        result = regrid(self.unstructured_grid_cube,
                        self.grid_for_unstructured_nearest,
                        'unstructured_nearest')
        expected = np.array([[[1]], [[1]], [[1]]])
        np.testing.assert_array_equal(result.data, expected)

        # Make sure that output is a masked array with correct fill value
        # (= maximum int)
        np.testing.assert_allclose(result.data.fill_value,
                                   float(np.iinfo(int).max))
