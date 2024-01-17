"""
Unit tests for the
:func:`esmvalcore.preprocessor.regrid.extract_point` function.

"""

import unittest
from unittest import mock

import iris.coords
import iris.cube
import numpy as np
from iris.analysis.cartography import get_xy_grids, unrotate_pole
from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.tests.stock import lat_lon_cube

import tests
from esmvalcore.preprocessor import extract_point
from esmvalcore.preprocessor._regrid import POINT_INTERPOLATION_SCHEMES

# TODO:
# use these to test extract point


class Test(tests.Test):

    def setUp(self):
        # Use an Iris test cube with coordinates that have a coordinate
        # system, see the following issue for more details:
        # https://github.com/ESMValGroup/ESMValCore/issues/2177.
        self.src_cube = lat_lon_cube()
        self.rpole_cube = _stock_rpole_2d()
        self.lambert_cube = _stock_lambert_2d()
        self.lambert_cube_nocs = _stock_lambert_2d_no_cs()

        self.schemes = ["linear", "nearest"]

    def test_invalid_scheme__unknown(self):
        dummy = mock.sentinel.dummy
        emsg = "Unknown interpolation scheme, got 'non-existent'"
        with self.assertRaisesRegex(ValueError, emsg):
            extract_point(dummy, dummy, dummy, 'non-existent')

    def test_invalid_coord_sys(self):
        latitude = -90.
        longitude = 0.
        emsg = 'If no coordinate system on cube then ' + \
               'can only interpolate lat-lon grids'

        with self.assertRaisesRegex(ValueError, emsg):
            extract_point(self.lambert_cube_nocs, latitude, longitude,
                          self.schemes[0])

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

    def test_extract_point_rpole(self):
        latitude = 90.
        longitude = -180.
        for scheme in self.schemes:
            result = extract_point(self.rpole_cube, latitude, longitude,
                                   scheme)
            self._assert_coords_rpole(result, latitude, longitude)

    def test_extract_point_lambert(self):
        latitude = 90.
        longitude = -180.
        for scheme in self.schemes:
            result = extract_point(self.lambert_cube, latitude, longitude,
                                   scheme)
            self._assert_coords_lambert(result, latitude, longitude)

    def _assert_coords(self, cube, ref_lat, ref_lon):
        """For a 1D cube with a lat-lon coord system check that a 1x1 cube is
        returned and the points are at the correct location."""
        lat_points = cube.coord("latitude").points
        lon_points = cube.coord("longitude").points
        self.assertEqual(len(lat_points), 1)
        self.assertEqual(len(lon_points), 1)
        self.assertEqual(lat_points[0], ref_lat)
        self.assertEqual(lon_points[0], ref_lon)

    def _assert_coords_rpole(self, cube, ref_lat, ref_lon):
        """For a 1D cube with a rotated coord system check that a 1x1 cube is
        returned and the points are at the correct location Will need to
        generate the lat and lon points from the grid using unrotate pole and
        test with approx equal."""

        pole_lat = cube.coord_system().grid_north_pole_latitude
        pole_lon = cube.coord_system().grid_north_pole_longitude
        rotated_lons, rotated_lats = get_xy_grids(cube)
        tlons, tlats = unrotate_pole(rotated_lons, rotated_lats, pole_lon,
                                     pole_lat)
        self.assertEqual(len(tlats), 1)
        self.assertEqual(len(tlons), 1)
        self.assertEqual(tlats[0], ref_lat)
        self.assertEqual(tlons[0], ref_lon)

    def _assert_coords_lambert(self, cube, ref_lat, ref_lon):
        """For a 1D cube with a Lambert coord system check that a 1x1 cube is
        returned and the points are at the correct location Will need to
        generate the lat and lon points from the grid."""

        pass


def _stock_rpole_2d():
    """Returns a realistic rotated pole 2d cube."""
    data = np.arange(9 * 11).reshape((9, 11))
    lat_pts = np.linspace(-4, 4, 9)
    lon_pts = np.linspace(-5, 5, 11)
    ll_cs = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))

    lat = iris.coords.DimCoord(
        lat_pts,
        standard_name="grid_latitude",
        units="degrees",
        coord_system=ll_cs,
    )
    lon = iris.coords.DimCoord(
        lon_pts,
        standard_name="grid_longitude",
        units="degrees",
        coord_system=ll_cs,
    )
    cube = iris.cube.Cube(
        data,
        standard_name="air_potential_temperature",
        units="K",
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
        attributes={"source": "test cube"},
    )
    return cube


def _stock_lambert_2d():
    """Returns a realistic lambert conformal 2d cube."""
    data = np.arange(9 * 11).reshape((9, 11))
    y_pts = np.linspace(-96000., 96000., 9)
    x_pts = np.linspace(0., 120000., 11)
    lam_cs = iris.coord_systems.LambertConformal(central_lat=48.0,
                                                 central_lon=9.75,
                                                 false_easting=-6000.0,
                                                 false_northing=-6000.0,
                                                 secant_latitudes=(30.0, 65.0),
                                                 ellipsoid=GeogCS(6371229.0))

    ydim = iris.coords.DimCoord(
        y_pts,
        standard_name="projection_y_coordinate",
        units="m",
        coord_system=lam_cs,
    )
    xdim = iris.coords.DimCoord(
        x_pts,
        standard_name="projection_x_coordinate",
        units="m",
        coord_system=lam_cs,
    )
    cube = iris.cube.Cube(
        data,
        standard_name="air_potential_temperature",
        units="K",
        dim_coords_and_dims=[(ydim, 0), (xdim, 1)],
        attributes={"source": "test cube"},
    )
    return cube


def _stock_lambert_2d_no_cs():
    """Returns a realistic lambert conformal 2d cube."""
    data = np.arange(9 * 11).reshape((9, 11))
    y_pts = np.linspace(-96000., 96000., 9)
    x_pts = np.linspace(0., 120000., 11)

    ydim = iris.coords.DimCoord(
        y_pts,
        standard_name="projection_y_coordinate",
        units="m",
    )
    xdim = iris.coords.DimCoord(
        x_pts,
        standard_name="projection_x_coordinate",
        units="m",
    )
    cube = iris.cube.Cube(
        data,
        standard_name="air_potential_temperature",
        units="K",
        dim_coords_and_dims=[(ydim, 0), (xdim, 1)],
        attributes={"source": "test cube"},
    )
    return cube


if __name__ == '__main__':
    unittest.main()
