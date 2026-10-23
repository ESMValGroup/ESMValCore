"""Unit tests for :func:`esmvalcore.preprocessor.extract_point`."""

import unittest
from unittest import mock

import numpy as np
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube

import tests
from esmvalcore.preprocessor import extract_point
from esmvalcore.preprocessor._regrid import POINT_INTERPOLATION_SCHEMES


def lat_lon_cube():
    """Create a cube with latitude and longitude coordinates.

    Copied from :func:`iris.tests.stock.lat_lon_cube` to avoid importing
    :mod:`iris.tests`, which requires internet access.
    """
    cs = GeogCS(6371229)
    lat = DimCoord(
        np.array([-1, 0, 1], dtype=np.int32),
        standard_name="latitude",
        units="degrees",
        coord_system=cs,
    )
    lon = DimCoord(
        np.array([-1, 0, 1, 2], dtype=np.int32),
        standard_name="longitude",
        units="degrees",
        coord_system=cs,
    )
    return Cube(
        np.arange(12, dtype=np.int32).reshape((3, 4)),
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )


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
            extract_point(dummy, dummy, dummy, "non-existent")

    def test_interpolation_schemes(self):
        self.assertEqual(
            set(POINT_INTERPOLATION_SCHEMES.keys()),
            set(self.schemes),
        )

    def test_extract_point_interpolation_schemes(self):
        latitude = -90.0
        longitude = 0.0
        for scheme in self.schemes:
            result = extract_point(self.src_cube, latitude, longitude, scheme)
            self._assert_coords(result, latitude, longitude)

    def test_extract_point(self):
        latitude = 90.0
        longitude = -180.0
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


if __name__ == "__main__":
    unittest.main()
