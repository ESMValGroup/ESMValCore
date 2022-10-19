"""Unit tests for :func:`esmvalcore.preprocessor.regrid.extract_levels`."""
import os
import tempfile
import unittest
from unittest import mock

import iris
import numpy as np
from numpy import ma

import tests
from esmvalcore.preprocessor._regrid import (
    _MDI,
    VERTICAL_SCHEMES,
    _preserve_fx_vars,
    extract_levels,
    parse_vertical_scheme,
)
from tests.unit.preprocessor._regrid import _make_cube, _make_vcoord


class Test(tests.Test):

    def setUp(self):
        self.shape = (3, 2, 1)
        self.z = self.shape[0]
        self.dtype = np.dtype('int8')
        data = np.arange(np.prod(self.shape),
                         dtype=self.dtype).reshape(self.shape)
        self.cube = _make_cube(data, dtype=self.dtype)
        self.created_cube = mock.sentinel.created_cube
        self.mock_create_cube = self.patch(
            'esmvalcore.preprocessor._regrid._create_cube',
            return_value=self.created_cube)
        self.schemes = [
            'linear', 'nearest', 'linear_extrapolate', 'nearest_extrapolate',
        ]
        descriptor, self.filename = tempfile.mkstemp('.nc')
        os.close(descriptor)

    def test_invalid_scheme__unknown(self):
        levels = mock.sentinel.levels
        scheme = mock.sentinel.scheme
        emsg = 'Unknown vertical interpolation scheme'
        with self.assertRaisesRegex(ValueError, emsg):
            extract_levels(self.cube, levels, scheme)

    def test_vertical_schemes(self):
        self.assertEqual(set(VERTICAL_SCHEMES), set(self.schemes))

    def test_parse_vertical_schemes(self):
        reference = {
            'linear': ('linear', 'nan'),
            'nearest': ('nearest', 'nan'),
            'linear_extrapolate': ('linear', 'nearest'),
            'nearest_extrapolate': ('nearest', 'nearest'),
        }
        for scheme in self.schemes:
            interpolation, extrapolation = parse_vertical_scheme(scheme)
            assert interpolation, extrapolation == reference[scheme]

    def test_nop__levels_match(self):
        vcoord = _make_vcoord(self.z, dtype=self.dtype)
        self.assertEqual(self.cube.coord(axis='z', dim_coords=True), vcoord)
        levels = vcoord.points
        result = extract_levels(self.cube, levels, 'linear')
        self.assertEqual(id(result), id(self.cube))
        self.assertEqual(result, self.cube)

    def test_extraction(self):
        levels = [0, 2]
        result = extract_levels(self.cube, levels, 'linear')
        data = np.array([0, 1, 4, 5], dtype=self.dtype).reshape(2, 2, 1)
        expected = _make_cube(data,
                              aux_coord=False,
                              dim_coord=False,
                              dtype=self.dtype)
        coord = self.cube.coord('Pressure Slice').copy()
        expected.add_aux_coord(coord[levels], (0, 1))
        coord = self.cube.coord('air_pressure').copy()
        expected.add_dim_coord(coord[levels], 0)
        self.assertEqual(result, expected)

    def test_fx_extraction(self):
        levels = [0, 2]
        area_data = np.ones((2, 1))
        volume_data = np.ones(self.shape)
        area_measure = iris.coords.CellMeasure(area_data,
                                               standard_name='cell_area',
                                               var_name='areacella',
                                               units='m2',
                                               measure='area')
        volume_measure = iris.coords.CellMeasure(volume_data,
                                                 standard_name='ocean_volume',
                                                 var_name='volcello',
                                                 units='m3',
                                                 measure='volume')
        ancillary_2d = iris.coords.AncillaryVariable(
            area_data,
            standard_name='land_area_fraction',
            var_name='sftlf',
            units='%')
        ancillary_3d = iris.coords.AncillaryVariable(
            volume_data,
            standard_name='height_above_reference_ellipsoid',
            var_name='zfull',
            units='m')
        self.cube.add_cell_measure(area_measure, (1, 2))
        self.cube.add_cell_measure(volume_measure, (0, 1, 2))
        self.cube.add_ancillary_variable(ancillary_2d, (1, 2))
        self.cube.add_ancillary_variable(ancillary_3d, (0, 1, 2))

        result = extract_levels(self.cube, levels, 'linear')

        data = np.array([0, 1, 4, 5], dtype=self.dtype).reshape(2, 2, 1)
        expected = _make_cube(data,
                              aux_coord=False,
                              dim_coord=False,
                              dtype=self.dtype)
        coord = self.cube.coord('Pressure Slice').copy()
        expected.add_aux_coord(coord[levels], (0, 1))
        coord = self.cube.coord('air_pressure').copy()
        expected.add_dim_coord(coord[levels], 0)
        expected.add_cell_measure(area_measure, (1, 2))
        expected.add_ancillary_variable(ancillary_2d, (1, 2))
        expected.add_cell_measure(volume_measure[0:2, ...], (0, 1, 2))
        expected.add_ancillary_variable(ancillary_3d[0:2, ...], (0, 1, 2))

        self.assertEqual(result, expected)

    def test_extraction__failure(self):
        levels = [0, 2]
        with mock.patch('iris.cube.Cube.extract', return_value=None):
            emsg = 'Failed to extract levels'
            with self.assertRaisesRegex(ValueError, emsg):
                extract_levels(self.cube, levels, 'linear')

    def test_interpolation(self):
        new_data = np.array(True)
        levels = np.array([0.5, 1.5])
        scheme = 'linear'
        with mock.patch('stratify.interpolate',
                        return_value=new_data) as mocker:
            result = extract_levels(self.cube, levels, scheme)
            self.assertEqual(result, self.created_cube)
            args, kwargs = mocker.call_args
            # Check the stratify.interpolate args ...
            self.assertEqual(len(args), 3)
            self.assert_array_equal(args[0], levels)
            pts = self.cube.coord(axis='z', dim_coords=True).points
            src_levels_broadcast = np.broadcast_to(pts.reshape(self.z, 1, 1),
                                                   self.cube.shape)
            self.assert_array_equal(args[1], src_levels_broadcast)
            self.assert_array_equal(args[2], self.cube.data)
            # Check the stratify.interpolate kwargs ...
            self.assertEqual(
                kwargs, dict(axis=0, interpolation=scheme,
                             extrapolation='nan'))
        args, kwargs = self.mock_create_cube.call_args
        # Check the _create_cube args ...
        self.assertEqual(len(args), 4)
        self.assertEqual(args[0], self.cube)
        self.assert_array_equal(args[1], new_data)
        self.assert_array_equal(args[2],
                                self.cube.coord(axis='z', dim_coords=True))
        self.assert_array_equal(args[3], levels)
        # Check the _create_cube kwargs ...
        self.assertEqual(kwargs, dict())

    def test_preserve_2d_fx_interpolation(self):
        area_data = np.ones((2, 1))
        area_measure = iris.coords.CellMeasure(area_data,
                                               standard_name='cell_area',
                                               var_name='areacella',
                                               units='m2',
                                               measure='area')
        ancillary_2d = iris.coords.AncillaryVariable(
            area_data,
            standard_name='land_area_fraction',
            var_name='sftlf',
            units='%')
        self.cube.add_cell_measure(area_measure, (1, 2))
        self.cube.add_ancillary_variable(ancillary_2d, (1, 2))
        result_data = np.array([0, 1, 4, 5], dtype=self.dtype).reshape(2, 2, 1)
        result = _make_cube(result_data)
        _preserve_fx_vars(self.cube, result)
        self.assertEqual(self.cube.cell_measures(), result.cell_measures())
        self.assertEqual(self.cube.ancillary_variables(),
                         result.ancillary_variables())

    def test_preserve_2d_fx_interpolation_single_level(self):
        result = self.cube[0, :, :]
        area_data = np.ones((2, 1))
        area_measure = iris.coords.CellMeasure(area_data,
                                               standard_name='cell_area',
                                               var_name='areacella',
                                               units='m2',
                                               measure='area')
        ancillary_2d = iris.coords.AncillaryVariable(
            area_data,
            standard_name='land_area_fraction',
            var_name='sftlf',
            units='%')
        self.cube.add_cell_measure(area_measure, (1, 2))
        self.cube.add_ancillary_variable(ancillary_2d, (1, 2))
        _preserve_fx_vars(self.cube, result)
        self.assertEqual(self.cube.cell_measures(), result.cell_measures())
        self.assertEqual(self.cube.ancillary_variables(),
                         result.ancillary_variables())

    def test_do_not_preserve_3d_fx_interpolation(self):
        volume_data = np.ones(self.shape)
        volume_measure = iris.coords.CellMeasure(volume_data,
                                                 standard_name='ocean_volume',
                                                 var_name='volcello',
                                                 units='m3',
                                                 measure='volume')
        ancillary_3d = iris.coords.AncillaryVariable(
            volume_data,
            standard_name='height_above_reference_ellipsoid',
            var_name='zfull',
            units='m')
        self.cube.add_cell_measure(volume_measure, (0, 1, 2))
        self.cube.add_ancillary_variable(ancillary_3d, (0, 1, 2))
        result_data = np.array([0, 1, 4, 5], dtype=self.dtype).reshape(2, 2, 1)
        result = _make_cube(result_data)
        with self.assertLogs(level='WARNING') as cm:
            _preserve_fx_vars(self.cube, result)
        self.assertEqual(
            cm.records[0].getMessage(),
            'Discarding use of z-axis dependent cell measure '
            'volcello in variable ta, as z-axis has been interpolated')
        self.assertEqual(
            cm.records[1].getMessage(),
            'Discarding use of z-axis dependent ancillary variable '
            'zfull in variable ta, as z-axis has been interpolated')

    def test_interpolation__extrapolated_nan_filling(self):
        new_data = np.array([0, np.nan])
        levels = [0.5, 1.5]
        scheme = 'nearest'
        with mock.patch('stratify.interpolate',
                        return_value=new_data) as mocker:
            result = extract_levels(self.cube, levels, scheme)
            self.assertEqual(result, self.created_cube)
            args, kwargs = mocker.call_args
            # Check the stratify.interpolate args ...
            self.assertEqual(len(args), 3)
            self.assert_array_equal(args[0], levels)
            pts = self.cube.coord(axis='z', dim_coords=True).points
            src_levels_broadcast = np.broadcast_to(pts.reshape(self.z, 1, 1),
                                                   self.cube.shape)
            self.assert_array_equal(args[1], src_levels_broadcast)
            self.assert_array_equal(args[2], self.cube.data)
            # Check the stratify.interpolate kwargs ...
            self.assertEqual(
                kwargs, dict(axis=0, interpolation=scheme,
                             extrapolation='nan'))
        args, kwargs = self.mock_create_cube.call_args
        # Check the _create_cube args ...
        self.assertEqual(len(args), 4)
        self.assert_array_equal(args[0], self.cube)
        new_data[np.isnan(new_data)] = _MDI
        new_data_mask = np.zeros(new_data.shape, bool)
        new_data_mask[new_data == _MDI] = True
        new_data = np.ma.array(new_data, mask=new_data_mask)
        self.assert_array_equal(args[1], new_data)
        self.assert_array_equal(args[2],
                                self.cube.coord(axis='z', dim_coords=True))
        self.assert_array_equal(args[3], levels)
        # Check the _create_cube kwargs ...
        self.assertEqual(kwargs, dict())

    def test_interpolation__masked(self):
        levels = np.array([0.5, 1.5])
        new_data = np.empty([len(levels)] + list(self.shape[1:]), dtype=float)
        new_data[:, 0, :] = np.nan
        new_data_mask = np.isnan(new_data)
        scheme = 'linear'
        mask = [[[False], [True]], [[True], [False]], [[False], [False]]]
        masked = ma.empty(self.shape)
        masked.mask = mask
        cube = _make_cube(masked, dtype=self.dtype)
        # save cube to test the lazy data interpolation too
        iris.save(cube, self.filename)
        with mock.patch('stratify.interpolate',
                        return_value=new_data) as mocker:
            # first test lazy
            loaded_cube = iris.load_cube(self.filename)
            result_from_lazy = extract_levels(loaded_cube, levels, scheme)
            self.assertEqual(result_from_lazy, self.created_cube)
            # then test realized
            result = extract_levels(cube, levels, scheme)
            self.assertEqual(result, self.created_cube)
            args, kwargs = mocker.call_args
            # Check the stratify.interpolate args ...
            self.assertEqual(len(args), 3)
            self.assert_array_equal(args[0], levels)
            pts = cube.coord(axis='z', dim_coords=True).points
            src_levels_broadcast = np.broadcast_to(pts.reshape(self.z, 1, 1),
                                                   cube.shape)
            self.assert_array_equal(args[1], src_levels_broadcast)
            self.assert_array_equal(args[2], cube.data)
            # Check the stratify.interpolate kwargs ...
            self.assertEqual(
                kwargs, dict(axis=0, interpolation=scheme,
                             extrapolation='nan'))
        args, kwargs = self.mock_create_cube.call_args
        input_cube = args[0]
        # in-place for new extract_levels with nan's
        new_data[np.isnan(new_data)] = _MDI
        # Check the _create_cube args ...
        self.assertEqual(len(args), 4)
        self.assertEqual(input_cube.metadata, cube.metadata)
        self.assertEqual(input_cube.coords, cube.coords)
        self.assertEqual(input_cube.coord_dims, cube.coord_dims)
        self.assert_array_equal(args[0].data, cube.data)
        new_data_mask = np.zeros(new_data.shape, bool)
        new_data_mask[new_data == _MDI] = True
        new_data = np.ma.array(new_data, mask=new_data_mask)
        self.assert_array_equal(args[1], new_data)
        self.assertTrue(ma.isMaskedArray(args[1]))
        self.assert_array_equal(args[1].mask, new_data_mask)
        self.assert_array_equal(args[2],
                                self.cube.coord(axis='z', dim_coords=True))
        self.assert_array_equal(args[3], levels)
        # Check the _create_cube kwargs ...
        self.assertEqual(kwargs, dict())


if __name__ == '__main__':
    unittest.main()
