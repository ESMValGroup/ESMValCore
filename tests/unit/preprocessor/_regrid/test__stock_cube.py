"""Unit tests for :func:`esmvalcore.preprocessor.regrid._stock_cube`."""

import unittest
from unittest import mock

import iris
import numpy as np

import tests
from esmvalcore.preprocessor._regrid import (
    _LAT_MAX,
    _LAT_MIN,
    _LAT_RANGE,
    _LON_MAX,
    _LON_MIN,
    _LON_RANGE,
    _global_stock_cube,
)


class Test(tests.Test):
    def _check(self, dx, dy, lat_off=True, lon_off=True):
        # Generate the expected stock cube coordinate points.
        dx, dy = float(dx), float(dy)
        mid_dx, mid_dy = dx / 2, dy / 2
        if lat_off and lon_off:
            expected_lat_points = np.linspace(
                _LAT_MIN + mid_dy,
                _LAT_MAX - mid_dy,
                int(_LAT_RANGE / dy),
            )
            expected_lon_points = np.linspace(
                _LON_MIN + mid_dx,
                _LON_MAX - mid_dx,
                int(_LON_RANGE / dx),
            )
        else:
            expected_lat_points = np.linspace(
                _LAT_MIN,
                _LAT_MAX,
                int(_LAT_RANGE / dy) + 1,
            )
            expected_lon_points = np.linspace(
                _LON_MIN,
                _LON_MAX - dx,
                int(_LON_RANGE / dx),
            )

        # Check the stock cube coordinates.
        self.assertEqual(self.mock_DimCoord.call_count, 2)
        call_lats, call_lons = self.mock_DimCoord.call_args_list

        # Check the latitude coordinate creation.
        [args], kwargs = call_lats
        self.assert_array_equal(args, expected_lat_points)
        expected_lat_kwargs = {
            "standard_name": "latitude",
            "units": "degrees_north",
            "var_name": "lat",
            "circular": False,
        }
        self.assertEqual(kwargs, expected_lat_kwargs)

        # Check the longitude coordinate creation.
        [args], kwargs = call_lons
        self.assert_array_equal(args, expected_lon_points)
        expected_lon_kwargs = {
            "standard_name": "longitude",
            "units": "degrees_east",
            "var_name": "lon",
            "circular": False,
        }
        self.assertEqual(kwargs, expected_lon_kwargs)

        # Check that the coordinate guess_bounds method has been called.
        expected_calls = [mock.call.guess_bounds()] * 2
        self.assertEqual(self.mock_coord.mock_calls, expected_calls)

        # Check the stock cube creation.
        self.mock_Cube.assert_called_once()
        _, kwargs = self.mock_Cube.call_args
        spec = [(self.mock_coord, 0), (self.mock_coord, 1)]
        expected_cube_kwargs = {"dim_coords_and_dims": spec}
        self.assertEqual(kwargs, expected_cube_kwargs)

        # Reset the mocks to enable multiple calls per test-case.
        for mocker in self.mocks:
            mocker.reset_mock()

    def setUp(self):
        self.Cube = mock.sentinel.Cube
        self.mock_Cube = self.patch(
            "esmvalcore.preprocessor._regrid.Cube",
            return_value=self.Cube,
        )
        self.mock_coord = mock.Mock(spec=iris.coords.DimCoord)
        self.mock_DimCoord = self.patch(
            "iris.coords.DimCoord",
            return_value=self.mock_coord,
        )
        self.mocks = [self.mock_Cube, self.mock_coord, self.mock_DimCoord]

    def tearDown(self) -> None:
        _global_stock_cube.cache_clear()
        return super().tearDown()

    def test_invalid_cell_spec__alpha(self):
        emsg = "Invalid MxN cell specification"
        with self.assertRaisesRegex(ValueError, emsg):
            _global_stock_cube("Ax1")

    def test_invalid_cell_spec__separator(self):
        emsg = "Invalid MxN cell specification"
        with self.assertRaisesRegex(ValueError, emsg):
            _global_stock_cube("1y1")

    def test_invalid_cell_spec__longitude(self):
        emsg = "Invalid longitude delta in MxN cell specification"
        with self.assertRaisesRegex(ValueError, emsg):
            _global_stock_cube("1.3x1")

    def test_invalid_cell_spec__latitude(self):
        emsg = "Invalid latitude delta in MxN cell specification"
        with self.assertRaisesRegex(ValueError, emsg):
            _global_stock_cube("1x2.3")

    def test_specs(self):
        specs = ["0.5x0.5", "1x1", "2.5x2.5", "5x5", "10x10"]
        for spec in specs:
            result = _global_stock_cube(spec)
            self.assertEqual(result, self.Cube)
            self._check(*list(map(float, spec.split("x"))))

    def test_specs_no_offset(self):
        specs = ["0.5x0.5", "1x1", "2.5x2.5", "5x5", "10x10"]
        for spec in specs:
            result = _global_stock_cube(
                spec,
                lat_offset=False,
                lon_offset=False,
            )
            self.assertEqual(result, self.Cube)
            self._check(
                *list(map(float, spec.split("x"))),
                lat_off=False,
                lon_off=False,
            )


if __name__ == "__main__":
    unittest.main()
