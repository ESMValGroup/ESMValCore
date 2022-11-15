"""Unit tests for the `esmvalcore.preprocessor._rolling_window` function."""
import unittest

import iris.coords
import iris.exceptions
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from numpy.testing import assert_equal

from esmvalcore.preprocessor._rolling_window import rolling_window_statistics


def _create_2d_cube():

    cube = Cube(np.broadcast_to(np.arange(1, 16), (11, 15)),
                var_name='tas',
                units='K')
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(-5, 6),
            standard_name='latitude',
            units=Unit('degrees'),
        ), 0)
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(1, 16),
            standard_name='time',
            units=Unit('days since 1950-01-01 00:00:00', calendar='gregorian'),
        ), 1)

    return cube


class TestRollingWindow(unittest.TestCase):
    """Test class for _rolling_window."""

    def setUp(self):
        """Prepare cube for tests."""
        self.cube = _create_2d_cube()

    def test_rolling_window_time(self):
        """Test rolling_window_statistics over time coordinate."""
        cube_time_sum = rolling_window_statistics(self.cube,
                                                  coordinate='time',
                                                  operator='sum',
                                                  window_length=2)
        expected_data = np.broadcast_to(np.arange(3, 30, 2), (11, 14))
        assert_equal(cube_time_sum.data, expected_data)
        assert cube_time_sum.shape == (11, 14)

    def test_rolling_window_latitude(self):
        """Test rolling_window_statistics over latitude coordinate."""
        cube_lat_mean = rolling_window_statistics(self.cube,
                                                  coordinate='latitude',
                                                  operator='mean',
                                                  window_length=3)
        expected_data = np.broadcast_to(np.arange(1, 16), (9, 15))
        assert_equal(cube_lat_mean.data, expected_data)
        assert cube_lat_mean.shape == (9, 15)

    def test_rolling_window_coord(self):
        self.cube.remove_coord('latitude')
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            rolling_window_statistics(self.cube,
                                      coordinate='latitude',
                                      operator='mean',
                                      window_length=3)

    def test_rolling_window_operator(self):
        with self.assertRaises(ValueError):
            rolling_window_statistics(self.cube,
                                      coordinate='time',
                                      operator='percentile',
                                      window_length=2)


if __name__ == '__main__':
    unittest.main()
