"""
Integration tests for the :func:`esmvalcore.preprocessor.regrid.regrid`
function.

"""

import unittest

import iris
import numpy as np
from numpy import ma

import tests
from esmvalcore.preprocessor import regrid
from tests.unit.preprocessor._regrid import _make_cube


class Test(tests.Test):
    def setUp(self):
        """Prepare tests."""
        shape = (3, 2, 2)
        data = np.arange(np.prod(shape)).reshape(shape)
        self.cube = _make_cube(data)
        self.cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

    def test_regrid__linear(self):
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
        result = regrid(self.cube, grid, 'linear')
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        self.assert_array_equal(result.data, expected)

    def test_regrid__linear_with_2d_aux_coord(self):
        data = np.empty((2, 2))
        lons = iris.coords.DimCoord([1.25, 1.75],
                                    standard_name='longitude',
                                    bounds=[[0.0, 1.5], [1.5, 2.0]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.25, 1.75],
                                    standard_name='latitude',
                                    bounds=[[0.0, 1.5], [1.5, 2.0]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)

        # Prepare cube
        aux_coord_bounds = iris.coords.AuxCoord(
            [[1, 3], [5, 7]],
            bounds=[[[0, 2], [2, 4]], [[4, 6], [6, 8]]],
            var_name='coord_with_bounds',
        )
        aux_coord_no_bounds = iris.coords.AuxCoord(
            [[10, 30], [50, 70]], var_name='coord_without_bounds',
        )
        cube = self.cube.copy()
        cube.add_aux_coord(aux_coord_bounds, (1, 2))
        cube.add_aux_coord(aux_coord_no_bounds, (1, 2))

        # Regridding
        result = regrid(cube, grid, 'linear')
        result_coord = result.coord('coord_without_bounds')
        expected_data = [[[0.75, 1.25],
                          [1.75, 2.25]],
                         [[4.75, 5.25],
                          [5.75, 6.25]],
                         [[8.75, 9.25],
                          [9.75, 10.25]]]
        expected_coord_points = [[25.0, 35.0],
                                 [45.0, 55.0]]
        self.assert_array_equal(result.data, expected_data)
        self.assert_array_equal(result_coord.points, expected_coord_points)
        assert result_coord.bounds is None
        assert not result.coords('coord_with_bounds')

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

    def test_regrid__nearest_with_1d_aux_coord(self):
        data = np.empty((2, 2))
        lons = iris.coords.DimCoord([1.25, 1.75],
                                    standard_name='longitude',
                                    bounds=[[0.0, 1.5], [1.5, 2.0]],
                                    units='degrees_east',
                                    coord_system=self.cs)
        lats = iris.coords.DimCoord([1.25, 1.75],
                                    standard_name='latitude',
                                    bounds=[[0.0, 1.5], [1.5, 2.0]],
                                    units='degrees_north',
                                    coord_system=self.cs)
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)

        # Prepare cube
        aux_coord = iris.coords.AuxCoord([1.0, 2.0], var_name='lat_2')
        cube = self.cube.copy()
        cube.add_aux_coord(aux_coord, 1)

        # Regridding
        result = regrid(cube, grid, 'nearest')
        result_coord = result.coord('lat_2')
        expected_data = [[[0.0, 1.0],
                          [2.0, 3.0]],
                         [[4.0, 5.0],
                          [6.0, 7.0]],
                         [[8.0, 9.0],
                          [10.0, 11.0]]]
        expected_coord_points = [1.25, 1.75]
        self.assert_array_equal(result.data, expected_data)
        self.assert_array_equal(result_coord.points, expected_coord_points)
        assert result_coord.bounds is None


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

    def test_regrid__unstructured_nearest(self):
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
        # Replace 1d spatial coords with 2d spatial coords.
        lons = self.cube.coord('longitude')
        lats = self.cube.coord('latitude')
        x, y = np.meshgrid(lons.points, lats.points)
        lats = iris.coords.AuxCoord(x, **lats._as_defn()._asdict())
        lons = iris.coords.AuxCoord(y, **lons._as_defn()._asdict())
        self.cube.remove_coord('longitude')
        self.cube.remove_coord('latitude')
        self.cube.remove_coord('Pressure Slice')
        self.cube.add_aux_coord(lons, (1, 2))
        self.cube.add_aux_coord(lats, (1, 2))
        result = regrid(self.cube, grid, 'unstructured_nearest')
        expected = np.array([[[3]], [[7]], [[11]]])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=6)


if __name__ == '__main__':
    unittest.main()
