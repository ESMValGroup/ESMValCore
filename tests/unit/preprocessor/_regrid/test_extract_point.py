"""
Unit tests for the
:func:`esmvalcore.preprocessor.regrid.extract_point` function.

"""

import unittest
from unittest import mock

import iris

import tests
from esmvalcore.preprocessor import extract_point
from esmvalcore.preprocessor._regrid import POINT_INTERPOLATION_SCHEMES


class Test(tests.Test):
    def setUp(self):
        self.coord_system = mock.Mock(return_value=None)
        self.coord = mock.sentinel.coord
        self.coords = mock.Mock(return_value=[self.coord])
        self.remove_coord = mock.Mock()
        self.point_cube = mock.sentinel.point_cube
        self.interpolate = mock.Mock(return_value=self.point_cube)
        self.src_cube = mock.Mock(
            spec=iris.cube.Cube,
            coord_system=self.coord_system,
            coords=self.coords,
            remove_coord=self.remove_coord,
            interpolate=self.interpolate)
        self.schemes = ['linear', 'nearest']

        self.mocks = [
            self.coord_system, self.coords, self.interpolate, self.src_cube
        ]

    def test_invalid_scheme__unknown(self):
        dummy = mock.sentinel.dummy
        emsg = "Unknown interpolation scheme, got 'non-existent'"
        with self.assertRaisesRegex(ValueError, emsg):
            extract_point(dummy, dummy, dummy, 'non-existent')

    def test_interpolation_schemes(self):
        self.assertEqual(
            set(POINT_INTERPOLATION_SCHEMES.keys()), set(self.schemes))

    def test_extract_point_interpolation_schemes(self):
        dummy = mock.sentinel.dummy
        for scheme in self.schemes:
            result = extract_point(self.src_cube, dummy, dummy, scheme)
            self.assertEqual(result, self.point_cube)

    def test_extract_point(self):
        dummy = mock.sentinel.dummy
        scheme = 'linear'
        result = extract_point(self.src_cube, dummy, dummy, scheme)
        self.assertEqual(result, self.point_cube)


if __name__ == '__main__':
    unittest.main()
