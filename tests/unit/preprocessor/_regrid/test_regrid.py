"""Unit tests for the :func:`esmvalcore.preprocessor.regrid.regrid`
function."""

import unittest
from unittest import mock

import iris
import numpy as np
import pytest

import tests
from esmvalcore.preprocessor import regrid
from esmvalcore.preprocessor._io import GLOBAL_FILL_VALUE
from esmvalcore.preprocessor._regrid import (
    _CACHE,
    HORIZONTAL_SCHEMES,
    _horizontal_grid_is_close,
)


class Test(tests.Test):
    def _check(self, tgt_grid, scheme, spec=False):
        expected_scheme = HORIZONTAL_SCHEMES[scheme]

        if spec:
            spec = tgt_grid
            self.assertIn(spec, _CACHE)
            self.assertEqual(_CACHE[spec], self.tgt_grid)
            self.coord_system.asset_called_once()
            expected_calls = [
                mock.call(axis='x', dim_coords=True),
                mock.call(axis='y', dim_coords=True)
            ]
            self.assertEqual(self.tgt_grid_coord.mock_calls, expected_calls)
            self.regrid.assert_called_once_with(self.tgt_grid, expected_scheme)
        else:
            if scheme == 'unstructured_nearest':
                expected_calls = [
                    mock.call(axis='x', dim_coords=True),
                    mock.call(axis='y', dim_coords=True)
                ]
                self.assertEqual(self.coords.mock_calls, expected_calls)
                expected_calls = [mock.call(self.coord), mock.call(self.coord)]
                self.assertEqual(self.remove_coord.mock_calls, expected_calls)
            self.regrid.assert_called_once_with(tgt_grid, expected_scheme)

        # Reset the mocks to enable multiple calls per test-case.
        for mocker in self.mocks:
            mocker.reset_mock()

    def setUp(self):
        self.coord_system = mock.Mock(return_value=None)
        self.coord = mock.sentinel.coord
        self.coords = mock.Mock(return_value=[self.coord])
        self.remove_coord = mock.Mock()
        self.regridded_cube = mock.Mock()
        self.regridded_cube.data = mock.sentinel.data
        self.regridded_cube_data = mock.Mock()
        self.regridded_cube.core_data.return_value = self.regridded_cube_data
        self.regrid = mock.Mock(return_value=self.regridded_cube)
        self.src_cube = mock.Mock(
            spec=iris.cube.Cube,
            coord_system=self.coord_system,
            coords=self.coords,
            remove_coord=self.remove_coord,
            regrid=self.regrid,
            dtype=float,
        )
        self.tgt_grid_coord = mock.Mock()
        self.tgt_grid = mock.Mock(
            spec=iris.cube.Cube, coord=self.tgt_grid_coord)
        self.regrid_schemes = [
            'linear', 'linear_extrapolate', 'nearest', 'area_weighted',
            'unstructured_nearest'
        ]

        def _mock_horizontal_grid_is_close(src, tgt):
            return False

        self.patch('esmvalcore.preprocessor._regrid._horizontal_grid_is_close',
                   side_effect=_mock_horizontal_grid_is_close)

        def _return_mock_global_stock_cube(
            spec,
            lat_offset=True,
            lon_offset=True,
        ):
            return self.tgt_grid

        self.mock_stock = self.patch(
            'esmvalcore.preprocessor._regrid._global_stock_cube',
            side_effect=_return_mock_global_stock_cube)
        self.mocks = [
            self.coord_system, self.coords, self.regrid, self.src_cube,
            self.tgt_grid_coord, self.tgt_grid, self.mock_stock
        ]

    def test_invalid_tgt_grid__unknown(self):
        dummy = mock.sentinel.dummy
        scheme = 'linear'
        emsg = 'Expecting a cube'
        with self.assertRaisesRegex(ValueError, emsg):
            regrid(self.src_cube, dummy, scheme)

    def test_invalid_scheme__unknown(self):
        dummy = mock.sentinel.dummy
        emsg = 'Unknown regridding scheme'
        with self.assertRaisesRegex(ValueError, emsg):
            regrid(dummy, dummy, 'wibble')

    def test_horizontal_schemes(self):
        self.assertEqual(
            set(HORIZONTAL_SCHEMES.keys()), set(self.regrid_schemes))

    @mock.patch('esmvalcore.preprocessor._regrid.da.ma.set_fill_value',
                autospec=True)
    @mock.patch('esmvalcore.preprocessor._regrid.da.ma.masked_equal',
                autospec=True)
    def test_regrid__horizontal_schemes(self, mock_masked_equal,
                                        mock_set_fill_value):
        mock_masked_equal.side_effect = lambda x, _: x
        for scheme in self.regrid_schemes:
            result = regrid(self.src_cube, self.tgt_grid, scheme)
            self.assertEqual(result, self.regridded_cube)

            # For 'unstructured_nearest', cube.data.astype() and
            # da.ma.masked_equal() are called
            if scheme == 'unstructured_nearest':
                mock_set_fill_value.assert_called_once()
                self.regridded_cube_data.astype.assert_called_once()
                mock_masked_equal.assert_called_once_with(
                    self.regridded_cube_data, GLOBAL_FILL_VALUE)
                self.assertEqual(result.data, self.regridded_cube_data)
            else:
                mock_set_fill_value.assert_not_called()
                self.regridded_cube_data.astype.assert_not_called()
                mock_masked_equal.assert_not_called()
                self.assertEqual(result.data, mock.sentinel.data)
            self._check(self.tgt_grid, scheme)

    def test_regrid__cell_specification(self):
        # Clear cache before and after the test to avoid poisoning
        # the cache with Mocked cubes
        # https://github.com/ESMValGroup/ESMValCore/issues/953
        _CACHE.clear()

        specs = ['1x1', '2x2', '3x3', '4x4', '5x5']
        scheme = 'linear'
        for spec in specs:
            result = regrid(self.src_cube, spec, scheme)
            self.assertEqual(result, self.regridded_cube)
            self.assertEqual(result.data, mock.sentinel.data)
            self._check(spec, scheme, spec=True)
        self.assertEqual(set(_CACHE.keys()), set(specs))

        _CACHE.clear()

    def test_regrid_generic_missing_reference(self):
        emsg = "No reference specified for generic regridding."
        with self.assertRaisesRegex(ValueError, emsg):
            regrid(self.src_cube, '1x1', {})

    def test_regrid_generic_invalid_reference(self):
        emsg = "Could not import specified generic regridding module."
        with self.assertRaisesRegex(ValueError, emsg):
            regrid(self.src_cube, '1x1', {"reference": "this.does:not.exist"})

    def test_regrid_generic_regridding(self):
        regrid(self.src_cube, '1x1', {"reference": "iris.analysis:Linear"})

    third_party_regridder = mock.Mock()

    def test_regrid_generic_third_party(self):
        regrid(self.src_cube, '1x1',
               {"reference":
                "tests.unit.preprocessor._regrid.test_regrid:"
                "Test.third_party_regridder",
                "method": "good",
                })
        self.third_party_regridder.assert_called_once_with(method="good")


def _make_coord(start: float, stop: float, step: int, *, name: str):
    """Helper function for creating a coord."""
    coord = iris.coords.DimCoord(
        np.linspace(start, stop, step),
        standard_name=name,
        units='degrees',
    )
    coord.guess_bounds()
    return coord


def _make_cube(*, lat: tuple, lon: tuple):
    """Helper function for creating a cube."""
    lat_coord = _make_coord(*lat, name='latitude')
    lon_coord = _make_coord(*lon, name='longitude')

    return iris.cube.Cube(
        np.empty([len(lat_coord.points),
                  len(lon_coord.points)]),
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
    )


# 10x10
LAT_SPEC1 = (-85, 85, 18)
LON_SPEC1 = (5, 355, 36)

# almost 10x10, but different shape
LAT_SPEC2 = (-85, 85, 17)
LON_SPEC2 = (5, 355, 35)

# 10x10, but different coords
LAT_SPEC3 = (-90, 90, 18)
LON_SPEC3 = (0, 360, 36)


@pytest.mark.parametrize(
    'cube2_spec, expected',
    (
        # equal lat/lon
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC1,
            },
            True,
        ),
        # different lon shape
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC2,
            },
            False,
        ),
        # different lat shape
        (
            {
                'lat': LAT_SPEC2,
                'lon': LON_SPEC1,
            },
            False,
        ),
        # different lon values
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC3,
            },
            False,
        ),
        # different lat values
        (
            {
                'lat': LAT_SPEC3,
                'lon': LON_SPEC1,
            },
            False,
        ),
    ),
)
def test_horizontal_grid_is_close(cube2_spec: dict, expected: bool):
    """Test for `_horizontal_grid_is_close`."""
    cube1 = _make_cube(lat=LAT_SPEC1, lon=LON_SPEC1)
    cube2 = _make_cube(**cube2_spec)

    assert _horizontal_grid_is_close(cube1, cube2) == expected


def test_regrid_is_skipped_if_grids_are_the_same():
    """Test that regridding is skipped if the grids are the same."""
    cube = _make_cube(lat=LAT_SPEC1, lon=LON_SPEC1)
    scheme = 'linear'

    # regridding to the same spec returns the same cube
    expected_same_cube = regrid(cube, target_grid='10x10', scheme=scheme)
    assert expected_same_cube is cube

    # regridding to a different spec returns a different cube
    expected_different_cube = regrid(cube, target_grid='5x5', scheme=scheme)
    assert expected_different_cube is not cube


if __name__ == '__main__':
    unittest.main()
