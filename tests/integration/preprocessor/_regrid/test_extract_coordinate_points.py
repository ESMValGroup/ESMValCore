"""
Integration tests for the :func:`esmvalcore.preprocessor.regrid.regrid`
function.

"""

import unittest

import iris
import numpy as np

import tests
from esmvalcore.preprocessor import extract_coordinate_points
from tests.unit.preprocessor._regrid import _make_cube


class Test(tests.Test):
    def setUp(self):
        """Prepare tests."""
        shape = (3, 4, 4)
        data = np.arange(np.prod(shape)).reshape(shape)
        self.cube = _make_cube(data, dtype=np.float64, rotated=True)
        self.cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

    def test_extract_point__single_linear(self):
        """Test linear interpolation when extracting a single point"""
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 2.1, 'grid_longitude': 2.1},
            scheme='linear')
        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(point.data, [5.5, 21.5, 37.5])

        # Exactly centred between grid points.
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 2.5, 'grid_longitude': 2.5},
            scheme='linear')
        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(point.data, [7.5, 23.5, 39.5])

        # On a (edge) grid point.
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 4, 'grid_longitude': 4},
            scheme='linear')
        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(point.data, [15, 31, 47])

        # Test two points outside the valid area.
        # These should be masked, since we set up the interpolation
        # schemes that way.
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': -1, 'grid_longitude': -1},
            scheme='linear')
        self.assertEqual(point.shape, (3,))
        masked = np.ma.array([np.nan] * 3, mask=True)
        self.assert_array_equal(point.data, masked)

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 30, 'grid_longitude': 30},
            scheme='linear')
        self.assertEqual(point.shape, (3,))
        self.assert_array_equal(point.data, masked)

    def test_extract_point__single_nearest(self):
        """Test nearest match when extracting a single point"""

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 2.1, 'grid_longitude': 2.1},
            scheme='nearest')
        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(point.data, [5, 21, 37])

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 4, 'grid_longitude': 4},
            scheme='nearest')
        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(point.data, [15, 31, 47])

        # Test two points outside the valid area
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': -1, 'grid_longitude': -1},
            scheme='nearest')
        self.assertEqual(point.shape, (3,))
        masked = np.ma.array(np.empty(3, dtype=np.float64), mask=True)
        self.assert_array_equal(point.data, masked)

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 30, 'grid_longitude': 30},
            scheme='nearest')
        self.assertEqual(point.shape, (3,))
        self.assert_array_equal(point.data, masked)

    def test_extract_point__multiple_linear(self):
        """Test linear interpolation for an array of one coordinate"""

        # Test points on the grid edges, on a grid point, halfway and
        # one in between.
        coords = self.cube.coords(dim_coords=True)
        print([coord.standard_name for coord in coords])

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [1, 1.1, 1.5, 2, 4],
             'grid_longitude': 2},
            scheme='linear')
        self.assertEqual(point.shape, (3, 5))
        # Longitude is not a dimension coordinate anymore.
        self.assertEqual(['air_pressure', 'grid_latitude'], [
            coord.standard_name for coord in point.coords(dim_coords=True)])
        np.testing.assert_allclose(point.data, [[1, 1.4, 3, 5, 13],
                                                [17, 17.4, 19., 21., 29],
                                                [33, 33.4, 35, 37, 45]])

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 4,
             'grid_longitude': [1, 1.1, 1.5, 2, 4]},
            scheme='linear')
        self.assertEqual(point.shape, (3, 5))
        self.assertEqual(['air_pressure', 'grid_longitude'], [
            coord.standard_name for coord in point.coords(dim_coords=True)])
        np.testing.assert_allclose(point.data, [[12, 12.1, 12.5, 13, 15],
                                                [28, 28.1, 28.5, 29, 31],
                                                [44, 44.1, 44.5, 45, 47]])

        # Test latitude and longitude points outside the grid.
        # These should all be masked.
        coords = self.cube.coords(dim_coords=True)
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [0, 10], 'grid_longitude': 3},
            scheme='linear')
        self.assertEqual(point.shape, (3, 2))
        masked = np.ma.array(np.empty((3, 2), dtype=np.float64), mask=True)
        self.assert_array_equal(point.data, masked)
        coords = self.cube.coords(dim_coords=True)
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 2, 'grid_longitude': [0, 10]},
            scheme='linear')
        coords = point.coords(dim_coords=True)
        self.assertEqual(point.shape, (3, 2))
        self.assert_array_equal(point.data, masked)

    def test_extract_point__multiple_nearest(self):
        """Test nearest match for an array of one coordinate"""

        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [1, 1.1, 1.5, 1.501, 2, 4],
             'grid_longitude': 2},
            scheme='nearest')
        self.assertEqual(point.shape, (3, 6))
        self.assertEqual(['air_pressure', 'grid_latitude'], [
            coord.standard_name for coord in point.coords(dim_coords=True)])
        np.testing.assert_allclose(point.data, [[1, 1, 1, 5, 5, 13],
                                                [17, 17, 17, 21, 21, 29],
                                                [33, 33, 33, 37, 37, 45]])
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 4,
             'grid_longitude': [1, 1.1, 1.5, 1.501, 2, 4]},
            scheme='nearest')
        self.assertEqual(point.shape, (3, 6))
        self.assertEqual(['air_pressure', 'grid_longitude'], [
            coord.standard_name for coord in point.coords(dim_coords=True)])
        np.testing.assert_allclose(point.data, [[12, 12, 12, 13, 13, 15],
                                                [28, 28, 28, 29, 29, 31],
                                                [44, 44, 44, 45, 45, 47]])
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [0, 10],
             'grid_longitude': 3},
            scheme='nearest')
        masked = np.ma.array(np.empty((3, 2), dtype=np.float64), mask=True)
        self.assertEqual(point.shape, (3, 2))
        self.assert_array_equal(point.data, masked)
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': 2,
             'grid_longitude': [0, 10]},
            scheme='nearest')
        self.assertEqual(point.shape, (3, 2))
        self.assert_array_equal(point.data, masked)

    def test_extract_point__multiple_both_linear(self):
        """Test for both latitude and longitude arrays, with
        linear interpolation"""
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [0, 1.1, 1.5, 1.51, 4, 5],
             'grid_longitude': [0, 1.1, 1.5, 1.51, 4, 5]}, scheme='linear')
        self.assertEqual(point.data.shape, (3, 6, 6))

        result = np.ma.array(np.empty((3, 6, 6), dtype=np.float64), mask=True)
        result[0, 1, 1:5] = [0.5, 0.9, 0.91, 3.4]
        result[0, 2, 1:5] = [2.1, 2.5, 2.51, 5.0]
        result[0, 3, 1:5] = [2.14, 2.54, 2.55, 5.04]
        result[0, 4, 1:5] = [12.1, 12.5, 12.51, 15.0]

        result[1, 1, 1:5] = [16.5, 16.9, 16.91, 19.4]
        result[1, 2, 1:5] = [18.10, 18.5, 18.51, 21.0]
        result[1, 3, 1:5] = [18.14, 18.54, 18.55, 21.04]
        result[1, 4, 1:5] = [28.1, 28.5, 28.51, 31.0]

        result[2, 1, 1:5] = [32.5, 32.9, 32.91, 35.4]
        result[2, 2, 1:5] = [34.1, 34.5, 34.51, 37]
        result[2, 3, 1:5] = [34.14, 34.54, 34.55, 37.04]
        result[2, 4, 1:5] = [44.1, 44.5, 44.51, 47]
        # Unmask the inner area of the result array.
        # The outer edges of the extracted points are outside the cube
        # grid, and should thus be masked.
        result.mask[:, 1:5, 1:5] = False

        np.testing.assert_allclose(point.data, result)

    def test_extract_point__multiple_both_nearest(self):
        """Test for both latitude and longitude arrays, with nearest match"""
        point = extract_coordinate_points(
            self.cube,
            {'grid_latitude': [0, 1.1, 1.5, 1.51, 4, 5],
             'grid_longitude': [0, 1.1, 1.5, 1.51, 4, 5]},
            scheme='nearest')
        self.assertEqual(point.data.shape, (3, 6, 6))

        result = np.ma.array(np.empty((3, 6, 6), dtype=np.float64), mask=True)
        result[0, 1, 1:5] = [0.0, 0.0, 1.0, 3.0]
        result[0, 2, 1:5] = [0.0, 0.0, 1.0, 3.0]
        result[0, 3, 1:5] = [4.0, 4.0, 5.0, 7.0]
        result[0, 4, 1:5] = [12.0, 12.0, 13.0, 15.0]

        result[1, 1, 1:5] = [16.0, 16.0, 17.0, 19.0]
        result[1, 2, 1:5] = [16.0, 16.0, 17.0, 19.0]
        result[1, 3, 1:5] = [20.0, 20.0, 21.0, 23.0]
        result[1, 4, 1:5] = [28.0, 28.0, 29.0, 31.0]

        result[2, 1, 1:5] = [32.0, 32.0, 33.0, 35.0]
        result[2, 2, 1:5] = [32.0, 32.0, 33.0, 35.0]
        result[2, 3, 1:5] = [36.0, 36.0, 37.0, 39.0]
        result[2, 4, 1:5] = [44.0, 44.0, 45.0, 47.0]
        result.mask[:, 1:5, 1:5] = False

        np.testing.assert_allclose(point.data, result)

    def test_wrong_interpolation_scheme(self):
        """Test wrong interpolation scheme raises error."""
        self.assertRaises(
            ValueError,
            extract_coordinate_points,
            self.cube, {'grid_latitude': 0.}, 'wrong')


if __name__ == '__main__':
    unittest.main()
