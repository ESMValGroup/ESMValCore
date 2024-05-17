"""Unit test for :func:`esmvalcore.preprocessor._volume`."""

import unittest

import iris
import iris.fileformats
import numpy as np
from cf_units import Unit
from iris.exceptions import CoordinateMultiDimError

import tests
from esmvalcore.preprocessor._volume import (
    _add_axis_stats_weights_coord,
    axis_statistics,
    calculate_volume,
    depth_integration,
    extract_trajectory,
    extract_transect,
    extract_volume,
    volume_statistics,
)


class Test(tests.Test):
    """Test class for _volume_pp."""

    def setUp(self):
        """Prepare tests."""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        data1 = np.ones((3, 2, 2))
        data2 = np.ma.ones((2, 3, 2, 2))
        data3 = np.ma.ones((4, 3, 2, 2))
        mask3 = np.full((4, 3, 2, 2), False)
        mask3[0, 0, 0, 0] = True
        data3 = np.ma.array(data3, mask=mask3)

        time = iris.coords.DimCoord([15, 45],
                                    standard_name='time',
                                    bounds=[[1., 30.], [30., 60.]],
                                    units=Unit('days since 1950-01-01',
                                               calendar='gregorian'))
        time2 = iris.coords.DimCoord([1., 2., 3., 4.],
                                     standard_name='time',
                                     bounds=[
                                         [0.5, 1.5],
                                         [1.5, 2.5],
                                         [2.5, 3.5],
                                         [3.5, 4.5],
                                     ],
                                     units=Unit('days since 1950-01-01',
                                                calendar='gregorian'))

        zcoord = iris.coords.DimCoord([0.5, 5., 50.],
                                      long_name='zcoord',
                                      bounds=[[0., 2.5], [2.5, 25.],
                                              [25., 250.]],
                                      units='m',
                                      attributes={'positive': 'down'})
        scoord = iris.coords.DimCoord([36., 36.5, 37.],
                                      long_name='ocean_sigma_coordinate',
                                      bounds=[[35.5, 36.25], [36.25, 36.75],
                                              [36.75, 37.5]],
                                      units='kg m-3',
                                      attributes={'positive': 'down'})
        zcoord_nobounds = iris.coords.DimCoord([0.5, 5., 50.],
                                               long_name='zcoord',
                                               units='m',
                                               attributes={'positive': 'down'})
        zcoord_4d = iris.coords.AuxCoord(
            np.broadcast_to([[[[0.5]], [[5.]], [[50.]]]], (2, 3, 2, 2)),
            long_name='zcoord',
            bounds=np.broadcast_to(
                [[[[[0., 2.5]]], [[[2.5, 25.]]], [[[25., 250.]]]]],
                (2, 3, 2, 2, 2),
            ),
            units='m',
            attributes={'positive': 'down'},
        )
        zcoord_3d_invalid_bounds = iris.coords.AuxCoord(
            np.broadcast_to([[[0.5]], [[5.]], [[50.]]], (3, 2, 2)),
            long_name='zcoord',
            bounds=np.broadcast_to(
                [[[[0., 2.5, 2.5, 3.]]],
                 [[[2.5, 25., 25., 30.]]],
                 [[[25., 250., 250., 300.]]]],
                (3, 2, 2, 4),
            ),
            units='m',
            attributes={'positive': 'down'},
        )
        lons2 = iris.coords.DimCoord([1.5, 2.5],
                                     standard_name='longitude',
                                     bounds=[[1., 2.], [2., 3.]],
                                     units='degrees_east',
                                     coord_system=coord_sys)
        lats2 = iris.coords.DimCoord([1.5, 2.5],
                                     standard_name='latitude',
                                     bounds=[[1., 2.], [2., 3.]],
                                     units='degrees_north',
                                     coord_system=coord_sys)

        lons2d = iris.coords.AuxCoord([[1.5, 2.5], [1.2, 2.7]],
                                      standard_name='longitude',
                                      units='degrees_east',
                                      coord_system=coord_sys)
        lats2d = iris.coords.AuxCoord([[1.5, 2.5], [1.2, 2.7]],
                                      standard_name='latitude',
                                      units='degrees_north',
                                      coord_system=coord_sys)

        coords_spec3 = [(zcoord, 0), (lats2, 1), (lons2, 2)]
        self.grid_3d = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec3)

        coords_spec4 = [(time, 0), (zcoord, 1), (lats2, 2), (lons2, 3)]
        self.grid_4d = iris.cube.Cube(
            data2,
            dim_coords_and_dims=coords_spec4,
            units='kg m-3',
        )

        coords_spec4_sigma = [(time, 0), (scoord, 1), (lats2, 2), (lons2, 3)]
        self.grid_4d_sigma_space = iris.cube.Cube(
            data2,
            dim_coords_and_dims=coords_spec4_sigma,
            units='kg m-3',
        )

        coords_spec5 = [(time2, 0), (zcoord, 1), (lats2, 2), (lons2, 3)]
        self.grid_4d_2 = iris.cube.Cube(
            data3,
            dim_coords_and_dims=coords_spec5,
            units='kg m-3',
        )

        self.grid_4d_z = iris.cube.Cube(
            data2,
            dim_coords_and_dims=[(time, 0), (lats2, 2), (lons2, 3)],
            aux_coords_and_dims=[(zcoord_4d, (0, 1, 2, 3))],
            units='kg m-3',
        )

        self.grid_4d_znobounds = iris.cube.Cube(
            data2,
            dim_coords_and_dims=[
                (time, 0), (zcoord_nobounds, 1), (lats2, 2), (lons2, 3)
            ],
            units='kg m-3',
        )

        self.grid_4d_irregular = iris.cube.Cube(
            data2,
            dim_coords_and_dims=[(time, 0), (zcoord, 1)],
            aux_coords_and_dims=[(lats2d, (2, 3)), (lons2d, (2, 3))],
            units='kg m-3',
        )

        self.grid_invalid_z_bounds = iris.cube.Cube(
            data2,
            dim_coords_and_dims=[(time, 0), (lats2, 2), (lons2, 3)],
            aux_coords_and_dims=[(zcoord_3d_invalid_bounds, (1, 2, 3))],
            units='kg m-3',
        )

        # allow iris to figure out the axis='z' coordinate
        iris.util.guess_coord_axis(self.grid_3d.coord('zcoord'))
        iris.util.guess_coord_axis(self.grid_4d.coord('zcoord'))
        iris.util.guess_coord_axis(self.grid_4d_2.coord('zcoord'))
        iris.util.guess_coord_axis(self.grid_4d_z.coord('zcoord'))

    def test_add_axis_stats_weights_coord(self):
        """Test _add_axis_stats_weights_coord."""
        assert not self.grid_4d.coords('_axis_statistics_weights_')
        coord = self.grid_4d.coord('zcoord')
        coord_dims = self.grid_4d.coord_dims('zcoord')
        _add_axis_stats_weights_coord(self.grid_4d, coord, coord_dims)
        weights_coord = self.grid_4d.coord('_axis_statistics_weights_')
        assert not weights_coord.has_lazy_points()
        assert weights_coord.units == 'm'
        np.testing.assert_allclose(weights_coord.points, [2.5, 22.5, 225.0])

    def test_add_axis_stats_weights_coord_lazy(self):
        """Test _add_axis_stats_weights_coord."""
        self.grid_4d.data = self.grid_4d.lazy_data()
        assert not self.grid_4d.coords('_axis_statistics_weights_')
        coord = self.grid_4d.coord('zcoord')
        coord_dims = self.grid_4d.coord_dims('zcoord')
        _add_axis_stats_weights_coord(self.grid_4d, coord, coord_dims)
        weights_coord = self.grid_4d.coord('_axis_statistics_weights_')
        assert weights_coord.has_lazy_points()
        assert weights_coord.units == 'm'
        np.testing.assert_allclose(weights_coord.points, [2.5, 22.5, 225.0])

    def test_axis_statistics_mean(self):
        """Test axis statistics with operator mean."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(self.grid_4d, 'z', 'mean')
        bounds = self.grid_4d.coord(axis='z').bounds
        weights = (bounds[:, 1] - bounds[:, 0])
        expected = np.average(data, axis=1, weights=weights)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')
        self.assertFalse(self.grid_4d.coords('_axis_statistics_weights_'))
        self.assertFalse(result.coords('_axis_statistics_weights_'))

    def test_axis_statistics_median(self):
        """Test axis statistics in with operator median."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(self.grid_4d, 'z', 'median')
        expected = np.median(data, axis=1)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_axis_statistics_min(self):
        """Test axis statistics with operator min."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(self.grid_4d, 'z', 'min')
        expected = np.min(data, axis=1)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_axis_statistics_max(self):
        """Test axis statistics with operator max."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(self.grid_4d, 'z', 'max')
        expected = np.max(data, axis=1)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_axis_statistics_rms(self):
        """Test axis statistics with operator rms."""
        result = axis_statistics(self.grid_4d, 'z', 'rms')
        expected = np.ma.ones((2, 2, 2))
        self.assert_array_equal(result.data, expected)

    def test_axis_statistics_std(self):
        """Test axis statistics with operator std_dev."""
        result = axis_statistics(self.grid_4d, 'z', 'std_dev')
        expected = np.ma.zeros((2, 2, 2))
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_axis_statistics_variance(self):
        """Test axis statistics with operator variance."""
        result = axis_statistics(self.grid_4d, 'z', 'variance')
        expected = np.ma.zeros((2, 2, 2))
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg2 m-6')

    def test_axis_statistics_sum(self):
        """Test axis statistics in multiple operators."""
        result = axis_statistics(self.grid_4d, 'z', 'sum')
        expected = np.ma.ones((2, 2, 2)) * 250
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-2')

    def test_axis_statistics_subtract_mean(self):
        """Test axis statistics with operator mean."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(
            self.grid_4d, 'z', 'mean', normalize='subtract'
        )
        bounds = self.grid_4d.coord(axis='z').bounds
        weights = (bounds[:, 1] - bounds[:, 0])
        expected = (
            data - np.average(data, axis=1, weights=weights, keepdims=True)
        )
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')
        self.assertFalse(self.grid_4d.coords('_axis_statistics_weights_'))
        self.assertFalse(result.coords('_axis_statistics_weights_'))

    def test_axis_statistics_divide_min(self):
        """Test axis statistics with operator sum."""
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        result = axis_statistics(self.grid_4d, 'z', 'min', normalize='divide')
        expected = data / np.min(data, axis=1, keepdims=True)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, '1')
        self.assertFalse(self.grid_4d.coords('_axis_statistics_weights_'))
        self.assertFalse(result.coords('_axis_statistics_weights_'))

    def test_wrong_axis_statistics_fail(self):
        """Test raises error when axis is not found in cube."""
        with self.assertRaises(ValueError) as err:
            axis_statistics(self.grid_3d, 't', 'mean')
        self.assertEqual(
            f'Axis t not found in cube {self.grid_3d.summary(shorten=True)}',
            str(err.exception))

    def test_multidimensional_axis_statistics_fail(self):
        i_coord = iris.coords.DimCoord(
            [0, 1],
            long_name='cell index along first dimension',
            units='1',
        )

        j_coord = iris.coords.DimCoord(
            [0, 1],
            long_name='cell index along second dimension',
            units='1',
        )

        lat_coord = iris.coords.AuxCoord(
            [[-40.0, -20.0], [-20.0, 0.0]],
            var_name='lat',
            standard_name='latitude',
            units='degrees_north',
        )

        lon_coord = iris.coords.AuxCoord(
            [[100.0, 140.0], [80.0, 100.0]],
            var_name='lon',
            standard_name='longitude',
            units='degrees_east',
        )

        cube = iris.cube.Cube(
            np.ones((2, 2)),
            var_name='tos',
            long_name='sea_surface_temperature',
            units='K',
            dim_coords_and_dims=[(j_coord, 0), (i_coord, 1)],
            aux_coords_and_dims=[(lat_coord, (0, 1)), (lon_coord, (0, 1))],
        )

        with self.assertRaises(NotImplementedError) as err:
            axis_statistics(cube, 'x', 'mean')
        self.assertEqual(('axis_statistics not implemented for '
                          'multidimensional coordinates.'), str(err.exception))

    def test_extract_volume(self):
        """Test to extract the top two layers of a 3 layer depth column."""
        result = extract_volume(self.grid_3d, 0., 10.)
        expected = np.ones((2, 2, 2))
        print(result.data, expected.data)
        self.assert_array_equal(result.data, expected)

    def test_extract_volume_intervals(self):
        """Test to extract open and closed intervals."""
        open_interval = extract_volume(self.grid_3d, 0., 5.)
        expected_levels_open = np.array([
            0.5,
        ])

        closed_interval = extract_volume(self.grid_3d, 0., 5., 'closed')
        expected_levels_closed = np.array([0.5, 5.])

        self.assert_array_equal(
            open_interval.coord(axis='Z').points, expected_levels_open)
        self.assert_array_equal(
            closed_interval.coord(axis='Z').points, expected_levels_closed)

    def test_extract_volume_mixed_intervals(self):
        left_closed = extract_volume(self.grid_3d, 0.5, 5., 'left_closed')
        expected_levels_left = np.array([
            0.5,
        ])

        right_closed = extract_volume(self.grid_3d, 0.5, 5., 'right_closed')
        expected_levels_right = np.array([
            5.,
        ])

        self.assert_array_equal(
            left_closed.coord(axis='Z').points, expected_levels_left)
        self.assert_array_equal(
            right_closed.coord(axis='Z').points, expected_levels_right)

    def test_extract_volume_nearest_values(self):
        """Test to extract nearest values."""
        default = extract_volume(self.grid_3d, 0, 48, 'closed', False)
        expected_levels_default = np.array([0.5, 5.])

        nearest = extract_volume(self.grid_3d, 0, 48, 'closed', True)
        expected_levels_nearest = np.array([0.5, 5., 50.])

        self.assert_array_equal(
            default.coord(axis='Z').points, expected_levels_default)
        self.assert_array_equal(
            nearest.coord(axis='Z').points, expected_levels_nearest)

    def test_extract_volume_error(self):
        with self.assertRaises(ValueError) as err:
            extract_volume(self.grid_3d, 0., 5., 'wrong')
        self.assertEqual(
            ('Depth extraction bounds can be set to "open", "closed", '
             '"left_closed", or "right_closed". '
             'Got "wrong".'), str(err.exception))

    def test_extract_volume_mean(self):
        """Test to extract the top two layers and compute the weighted average
        of a cube."""
        grid_volume = calculate_volume(self.grid_4d)
        measure = iris.coords.CellMeasure(grid_volume,
                                          standard_name='ocean_volume',
                                          units='m3',
                                          measure='volume')
        self.grid_4d.add_cell_measure(measure, range(0, measure.ndim))

        result = extract_volume(self.grid_4d, 0., 10.)

        expected = np.ma.ones((2, 2, 2, 2))
        self.assert_array_equal(result.data, expected)

        result_mean = volume_statistics(result, 'mean')

        expected_mean = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result_mean.data, expected_mean)
        self.assertEqual(result_mean.units, 'kg m-3')
        self.assertTrue(self.grid_4d.cell_measures('ocean_volume'))
        self.assertFalse(result_mean.cell_measures('ocean_volume'))

    def test_volume_statistics(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube."""
        self.assertFalse(self.grid_4d.cell_measures('ocean_volume'))

        result = volume_statistics(self.grid_4d, 'mean')

        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')
        self.assertFalse(self.grid_4d.cell_measures('ocean_volume'))
        self.assertFalse(result.cell_measures('ocean_volume'))

    def test_volume_nolevbounds(self):
        """Test to take the volume weighted average of a cube with no bounds
        in the z axis.
        """

        self.assertFalse(self.grid_4d_znobounds.coord(axis='z').has_bounds())
        result = volume_statistics(self.grid_4d_znobounds, 'mean')

        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')
        self.assertFalse(self.grid_4d.cell_measures('ocean_volume'))
        self.assertFalse(result.cell_measures('ocean_volume'))

    def test_volume_statistics_cell_measure(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube.

        The volume measure is pre-loaded in the cube.
        """
        grid_volume = calculate_volume(self.grid_4d)
        measure = iris.coords.CellMeasure(grid_volume,
                                          standard_name='ocean_volume',
                                          units='m3',
                                          measure='volume')
        self.grid_4d.add_cell_measure(measure, range(0, measure.ndim))
        result = volume_statistics(self.grid_4d, 'mean')
        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_volume_statistics_long(self):
        """Test to take the volume weighted average of a (4,3,2,2) cube.

        This extra time is needed, as the volume average calculation
        uses different methods for small and large cubes.
        """
        result = volume_statistics(self.grid_4d_2, 'mean')
        expected = np.ma.array([1., 1., 1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_volume_statistics_masked_level(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube where
        the last depth level is fully masked."""
        self.grid_4d.data[:, -1, :, :] = np.ma.masked_all((2, 2, 2))
        result = volume_statistics(self.grid_4d, 'mean')
        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)

    def test_volume_statistics_masked_timestep(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube where
        the first timestep is fully masked."""
        self.grid_4d.data[0, :, :, :] = np.ma.masked_all((3, 2, 2))
        result = volume_statistics(self.grid_4d, 'mean')
        expected = np.ma.array([1., 1], mask=[True, False])
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_volume_statistics_subtract_mean(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube."""
        self.assertFalse(self.grid_4d.cell_measures('ocean_volume'))

        result = volume_statistics(self.grid_4d, 'mean', normalize='subtract')

        expected = np.ma.zeros((2, 3, 2, 2))
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')
        self.assertFalse(self.grid_4d.cell_measures('ocean_volume'))
        self.assertFalse(result.cell_measures('ocean_volume'))

    def test_volume_statistics_cell_measure_divide_mean(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube.

        The volume measure is pre-loaded in the cube.
        """
        grid_volume = calculate_volume(self.grid_4d)
        measure = iris.coords.CellMeasure(grid_volume,
                                          standard_name='ocean_volume',
                                          units='m3',
                                          measure='volume')
        self.grid_4d.add_cell_measure(measure, range(0, measure.ndim))

        result = volume_statistics(self.grid_4d, 'mean', normalize='divide')

        expected = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, '1')

    def test_volume_statistics_weights(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube.

        The data and weights are not defined as arrays of ones.
        """
        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d.data = data
        measure = iris.coords.CellMeasure(data,
                                          standard_name='ocean_volume',
                                          units='m3',
                                          measure='volume')
        self.grid_4d.add_cell_measure(measure, range(0, measure.ndim))
        result = volume_statistics(self.grid_4d, 'mean')
        expected = np.ma.array([8.333333333333334, 19.144144144144143],
                               mask=[False, False])
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_volume_statistics_wrong_operator_fail(self):
        with self.assertRaises(ValueError) as err:
            volume_statistics(self.grid_4d, 'wrong')
        self.assertEqual('Volume operator wrong not recognised.',
                         str(err.exception))

    def test_volume_statistics_2d_lat_fail(self):
        with self.assertRaises(CoordinateMultiDimError):
            volume_statistics(self.grid_4d_irregular, 'mean')

    def test_volume_statistics_2d_lat_cellarea(self):
        measure = iris.coords.CellMeasure(np.arange(1, 5).reshape(2, 2),
                                          standard_name='cell_area',
                                          units='m2',
                                          measure='area')
        self.grid_4d_irregular.add_cell_measure(measure, (2, 3))

        result = volume_statistics(self.grid_4d_irregular, 'mean')
        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

        data = np.ma.arange(1, 25).reshape(2, 3, 2, 2)
        self.grid_4d_irregular.data = data

        result = volume_statistics(self.grid_4d_irregular, 'mean')
        expected = np.ma.array([10.56, 22.56], mask=False)
        self.assert_array_equal(result.data, expected)
        self.assertEqual(result.units, 'kg m-3')

    def test_volume_statistics_invalid_bounds(self):
        """Test z-axis bounds is not 2 in last dimension"""

        with self.assertRaises(ValueError) as err:
            volume_statistics(self.grid_invalid_z_bounds, 'mean')
        self.assertIn(
            "Z axis bounds shape found (3, 2, 2, 4). Bounds should be "
            "2 in the last dimension to compute the thickness.",
            str(err.exception)
        )

    def test_volume_statistics_invalid_units(self):
        """Test z-axis units cannot be converted to m"""

        with self.assertRaises(ValueError) as err:
            volume_statistics(self.grid_4d_sigma_space, 'mean')
        self.assertIn(
            "Cannot compute volume using the Z-axis. "
            "Unable to convert from 'Unit('kg m-3')' to 'Unit('m')'.",
            str(err.exception)
        )

    def test_volume_statistics_z_axis_time_error(self):
        # Fails because depth z-axis coord depends on time dimensions
        # which would aggregate also along that dimension
        with self.assertRaises(ValueError) as err:
            volume_statistics(self.grid_4d_z, 'mean')
        self.assertIn(
            "X and Y axis coordinates depend on (2, 3) dimensions, "
            "while X, Y, and Z axis depends on (0, 1, 2, 3) dimensions. "
            "This may indicate Z axis depending on other dimension than "
            "space that could provoke invalid aggregation...",
            str(err.exception)
        )

        grid_3d_no_x = self.grid_4d_z[..., 0]
        with self.assertRaises(ValueError) as err:
            volume_statistics(grid_3d_no_x, 'mean')
        self.assertIn(
            "X and Y axis coordinates depend on (2,) dimensions, "
            "while X, Y, and Z axis depends on (0, 1, 2) dimensions. "
            "This may indicate Z axis depending on other dimension than "
            "space that could provoke invalid aggregation...",
            str(err.exception)
        )

    def test_volume_statistics_missing_axis(self):
        # x axis is missing
        grid_no_x = self.grid_4d[..., 0]
        volume_statistics(grid_no_x, 'mean')

        # y axis is missing
        grid_no_y = self.grid_4d[..., 0, :]
        volume_statistics(grid_no_y, 'mean')

        # z axis is missing
        grid_no_z = self.grid_4d[:, 0]
        with self.assertRaises(ValueError) as err:
            volume_statistics(grid_no_z, 'mean')
        self.assertIn("Cannot compute volume with scalar Z-axis",
                      str(err.exception))

    def test_volume_statistics_2d_depth(self):
        # Create new 2D depth coord
        new_z_coord = self.grid_4d_z.coord('zcoord')[0, :, :, 0]
        self.grid_4d_z.remove_coord('zcoord')
        self.grid_4d_z.add_aux_coord(new_z_coord, (1, 2))
        result = volume_statistics(self.grid_4d, 'mean')
        expected = np.ma.array([1., 1.], mask=False)
        self.assert_array_equal(result.data, expected)

    def test_depth_integration_1d(self):
        """Test to take the depth integration of a 3 layer cube."""
        result = depth_integration(self.grid_3d[:, 0, 0])
        expected = np.ones((1, 1)) * 250.
        self.assert_array_equal(result.data, expected)

    def test_depth_integration_3d(self):
        """Test to take the depth integration of a 3 layer cube."""
        result = depth_integration(self.grid_3d)
        expected = np.ones((2, 2)) * 250.
        self.assert_array_equal(result.data, expected)

    def test_extract_transect_latitude(self):
        """Test to extract a transect from a (3, 2, 2) cube."""
        result = extract_transect(self.grid_3d, latitude=1.5)
        expected = np.ones((3, 2))
        self.assert_array_equal(result.data, expected)

    def test_extract_transect_longitude(self):
        """Test to extract a transect from a (3, 2, 2) cube."""
        result = extract_transect(self.grid_3d, longitude=1.5)
        expected = np.ones((3, 2))
        self.assert_array_equal(result.data, expected)

    def test_extract_trajectory(self):
        """Test to extract a trajectory from a (3, 2, 2) cube."""
        result = extract_trajectory(self.grid_3d, [1.5, 2.5], [2., 2.], 2)
        expected = np.ones((3, 2))
        self.assert_array_equal(result.data, expected)


if __name__ == '__main__':
    unittest.main()
