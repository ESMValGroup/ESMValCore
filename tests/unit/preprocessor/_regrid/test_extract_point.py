"""
Unit tests for the
:func:`esmvalcore.preprocessor.regrid.extract_point` function.

"""

import unittest
from unittest import mock

from iris.tests.stock import lat_lon_cube

import tests
from esmvalcore.preprocessor import extract_point
from esmvalcore.preprocessor._regrid import POINT_INTERPOLATION_SCHEMES


class Test(tests.Test):

    def setUp(self):
        # Use an Iris test cube with coordinates that have a coordinate
        # system, see the following issue for more details:
        # https://github.com/ESMValGroup/ESMValCore/issues/2177.
        self.src_cube = lat_lon_cube()
        self.schemes = ["linear", "nearest"]

    def test_invalid_scheme__unknown(self):
        dummy = mock.sentinel.dummy
        emsg = "Unknown interpolation scheme, got 'non-existent'"
        with self.assertRaisesRegex(ValueError, emsg):
            extract_point(dummy, dummy, dummy, 'non-existent')

    def test_interpolation_schemes(self):
        self.assertEqual(set(POINT_INTERPOLATION_SCHEMES.keys()),
                         set(self.schemes))

    def test_extract_point_interpolation_schemes(self):
        latitude = -90.
        longitude = 0.
        for scheme in self.schemes:
            result = extract_point(self.src_cube, latitude, longitude, scheme)
            self._assert_coords(result, latitude, longitude)

    def test_extract_point(self):
        latitude = 90.
        longitude = -180.
        for scheme in self.schemes:
            result = extract_point(self.src_cube, latitude, longitude, scheme)
            self._assert_coords(result, latitude, longitude)

    def _assert_coords(self, cube, ref_lat, ref_lon):
        lat_points = cube.coord("latitude").points
        lon_points = cube.coord("longitude").points
        self.assertEqual(len(lat_points), 1)
        self.assertEqual(len(lon_points), 1)
        self.assertEqual(lat_points[0], ref_lat)
        self.assertEqual(lon_points[0], ref_lon)


if __name__ == '__main__':
    unittest.main()
