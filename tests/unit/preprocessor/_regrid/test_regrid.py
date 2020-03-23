"""
Unit tests for the :func:`esmvalcore.preprocessor.regrid.regrid` function.

"""

import unittest
from unittest import mock

import iris
import numpy as np

import tests
from esmvalcore.preprocessor import regrid
from esmvalcore.preprocessor._regrid import _CACHE, HORIZONTAL_SCHEMES
from esmvalcore.preprocessor._regrid import _check_horiz_grid_closeness


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
        self.regridded_cube = mock.sentinel.regridded_cube
        self.regrid = mock.Mock(return_value=self.regridded_cube)
        self.src_cube = mock.Mock(
            spec=iris.cube.Cube,
            coord_system=self.coord_system,
            coords=self.coords,
            remove_coord=self.remove_coord,
            regrid=self.regrid)
        self.tgt_grid_coord = mock.Mock()
        self.tgt_grid = mock.Mock(
            spec=iris.cube.Cube, coord=self.tgt_grid_coord)
        self.regrid_schemes = [
            'linear', 'linear_extrapolate', 'nearest', 'area_weighted',
            'unstructured_nearest'
        ]

        def _mock_check_horiz_grid_closeness(src, tgt):
            return False

        self.patch(
            'esmvalcore.preprocessor._regrid._check_horiz_grid_closeness',
            side_effect=_mock_check_horiz_grid_closeness)

        def _return_mock_stock_cube(spec, lat_offset=True, lon_offset=True):
            return self.tgt_grid

        self.mock_stock = self.patch(
            'esmvalcore.preprocessor._regrid._stock_cube',
            side_effect=_return_mock_stock_cube)
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

    def test_regrid__horizontal_schemes(self):
        for scheme in self.regrid_schemes:
            result = regrid(self.src_cube, self.tgt_grid, scheme)
            self.assertEqual(result, self.regridded_cube)
            self._check(self.tgt_grid, scheme)

    def test_regrid__cell_specification(self):
        specs = ['1x1', '2x2', '3x3', '4x4', '5x5']
        scheme = 'linear'
        for spec in specs:
            result = regrid(self.src_cube, spec, scheme)
            self.assertEqual(result, self.regridded_cube)
            self._check(spec, scheme, spec=True)
        self.assertEqual(set(_CACHE.keys()), set(specs))


class TestCloseness(tests.Test):
    def test_regrid__closeness_cell_specification(self):
        latitude = iris.coords.DimCoord(np.linspace(-85, 85, 18),
                                        standard_name='latitude',
                                        units='degrees')
        longitude = iris.coords.DimCoord(np.linspace(5, 355, 36),
                                         standard_name='longitude',
                                         units='degrees')
        latitude2 = iris.coords.DimCoord(np.linspace(-85, 85, 17),
                                         standard_name='latitude',
                                         units='degrees')
        longitude2 = iris.coords.DimCoord(np.linspace(5, 355, 35),
                                          standard_name='longitude',
                                          units='degrees')
        latitude.guess_bounds()
        longitude.guess_bounds()
        loc_cube = iris.cube.Cube(np.empty([18, 36]),
                                  dim_coords_and_dims=[(latitude, 0),
                                                       (longitude, 1)],
                                  )
        tgt_cubes = [loc_cube,
                     iris.cube.Cube(np.empty([18, 36]),
                                    dim_coords_and_dims=[(latitude, 0),
                                                         (longitude, 1)],
                                    ),
                     iris.cube.Cube(np.empty([17, 36]),
                                    dim_coords_and_dims=[(latitude2, 0),
                                                         (longitude, 1)],
                                    ),
                     iris.cube.Cube(np.empty([18, 35]),
                                    dim_coords_and_dims=[(latitude, 0),
                                                         (longitude2, 1)],
                                    )
                     ]

        tgt_closeness = [True, True, False, False]
        for tgt in zip(tgt_cubes, tgt_closeness):
            self.assertEqual(_check_horiz_grid_closeness(loc_cube, tgt[0]),
                             tgt[1])


if __name__ == '__main__':
    unittest.main()
