"""Unit test for :func:`esmvalcore.preprocessor._multimodel`."""

import unittest

import cftime
import iris
import numpy as np
from cf_units import Unit

import tests
from esmvalcore.preprocessor import multi_model_statistics
from esmvalcore.preprocessor._multimodel import (_assemble_full_data,
                                                 _assemble_overlap_data,
                                                 _compute_statistic,
                                                 _datetime_to_int_days,
                                                 _get_overlap,
                                                 _get_time_offset,
                                                 _plev_fix,
                                                 _put_in_cube,
                                                 _slice_cube)


class Test(tests.Test):
    """Test class for preprocessor/_multimodel.py."""

    def setUp(self):
        """Prepare tests."""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        data2 = np.ma.ones((2, 3, 2, 2))
        data3 = np.ma.ones((4, 3, 2, 2))
        mask3 = np.full((4, 3, 2, 2), False)
        mask3[0, 0, 0, 0] = True
        data3 = np.ma.array(data3, mask=mask3)

        time = iris.coords.DimCoord([15, 45],
                                    standard_name='time',
                                    bounds=[[1., 30.], [30., 60.]],
                                    units=Unit(
                                        'days since 1950-01-01',
                                        calendar='gregorian'))
        time2 = iris.coords.DimCoord([1., 2., 3., 4.],
                                     standard_name='time',
                                     bounds=[
                                         [0.5, 1.5],
                                         [1.5, 2.5],
                                         [2.5, 3.5],
                                         [3.5, 4.5], ],
                                     units=Unit(
                                         'days since 1950-01-01',
                                         calendar='gregorian'))
        yr_time = iris.coords.DimCoord([15, 410],
                                       standard_name='time',
                                       bounds=[[1., 30.], [395., 425.]],
                                       units=Unit(
                                           'days since 1950-01-01',
                                           calendar='gregorian'))
        yr_time2 = iris.coords.DimCoord([1., 367., 733., 1099.],
                                        standard_name='time',
                                        bounds=[
                                            [0.5, 1.5],
                                            [366, 368],
                                            [732, 734],
                                            [1098, 1100], ],
                                        units=Unit(
                                            'days since 1950-01-01',
                                            calendar='gregorian'))
        zcoord = iris.coords.DimCoord([0.5, 5., 50.],
                                      standard_name='air_pressure',
                                      long_name='air_pressure',
                                      bounds=[[0., 2.5], [2.5, 25.],
                                              [25., 250.]],
                                      units='m',
                                      attributes={'positive': 'down'})
        lons = iris.coords.DimCoord([1.5, 2.5],
                                    standard_name='longitude',
                                    long_name='longitude',
                                    bounds=[[1., 2.], [2., 3.]],
                                    units='degrees_east',
                                    coord_system=coord_sys)
        lats = iris.coords.DimCoord([1.5, 2.5],
                                    standard_name='latitude',
                                    long_name='latitude',
                                    bounds=[[1., 2.], [2., 3.]],
                                    units='degrees_north',
                                    coord_system=coord_sys)

        coords_spec4 = [(time, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube1 = iris.cube.Cube(data2, dim_coords_and_dims=coords_spec4)

        coords_spec5 = [(time2, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube2 = iris.cube.Cube(data3, dim_coords_and_dims=coords_spec5)

        coords_spec4_yr = [(yr_time, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube1_yr = iris.cube.Cube(data2,
                                       dim_coords_and_dims=coords_spec4_yr)

        coords_spec5_yr = [(yr_time2, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube2_yr = iris.cube.Cube(data3,
                                       dim_coords_and_dims=coords_spec5_yr)

    def test_get_time_offset(self):
        """Test time unit."""
        result = _get_time_offset("days since 1950-01-01")
        expected = cftime.real_datetime(1950, 1, 1, 0, 0)
        np.testing.assert_equal(result, expected)

    def test_compute_statistic(self):
        """Test statistic."""
        data = [self.cube1.data[0], self.cube2.data[0]]
        stat_mean = _compute_statistic(data, "mean")
        stat_median = _compute_statistic(data, "median")
        expected_mean = np.ma.ones((3, 2, 2))
        expected_median = np.ma.ones((3, 2, 2))
        self.assert_array_equal(stat_mean, expected_mean)
        self.assert_array_equal(stat_median, expected_median)

    def test_compute_full_statistic_mon_cube(self):
        data = [self.cube1, self.cube2]
        stats = multi_model_statistics(data, 'full', ['mean'])
        expected_full_mean = np.ma.ones((2, 3, 2, 2))
        expected_full_mean.mask = np.zeros((2, 3, 2, 2))
        expected_full_mean.mask[1] = True
        self.assert_array_equal(stats['mean'].data, expected_full_mean)

    def test_compute_full_statistic_yr_cube(self):
        data = [self.cube1_yr, self.cube2_yr]
        stats = multi_model_statistics(data, 'full', ['mean'])
        expected_full_mean = np.ma.ones((4, 3, 2, 2))
        expected_full_mean.mask = np.zeros((4, 3, 2, 2))
        expected_full_mean.mask[2:4] = True
        self.assert_array_equal(stats['mean'].data, expected_full_mean)

    def test_compute_overlap_statistic_mon_cube(self):
        data = [self.cube1, self.cube1]
        stats = multi_model_statistics(data, 'overlap', ['mean'])
        expected_ovlap_mean = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(stats['mean'].data, expected_ovlap_mean)

    def test_compute_overlap_statistic_yr_cube(self):
        data = [self.cube1_yr, self.cube1_yr]
        stats = multi_model_statistics(data, 'overlap', ['mean'])
        expected_ovlap_mean = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(stats['mean'].data, expected_ovlap_mean)

    def test_compute_std(self):
        """Test statistic."""
        data = [self.cube1.data[0], self.cube2.data[0] * 2]
        stat = _compute_statistic(data, "std")
        expected = np.ma.ones((3, 2, 2)) * 0.5
        expected[0, 0, 0] = 0
        self.assert_array_equal(stat, expected)

    def test_compute_max(self):
        """Test statistic."""
        data = [self.cube1.data[0] * 0.5, self.cube2.data[0] * 2]
        stat = _compute_statistic(data, "max")
        expected = np.ma.ones((3, 2, 2)) * 2
        expected[0, 0, 0] = 0.5
        self.assert_array_equal(stat, expected)

    def test_compute_min(self):
        """Test statistic."""
        data = [self.cube1.data[0] * 0.5, self.cube2.data[0] * 2]
        stat = _compute_statistic(data, "min")
        expected = np.ma.ones((3, 2, 2)) * 0.5
        self.assert_array_equal(stat, expected)

    def test_compute_percentile(self):
        """Test statistic."""
        data = [self.cube1.data[0] * 0.5, self.cube2.data[0] * 2]
        stat = _compute_statistic(data, "p75")
        expected = np.ma.ones((3, 2, 2)) * 1.625
        expected[0, 0, 0] = 0.5
        self.assert_array_equal(stat, expected)

    def test_put_in_cube(self):
        """Test put in cube."""
        cube_data = np.ma.ones((2, 3, 2, 2))
        stat_cube = _put_in_cube(self.cube1, cube_data, "mean", t_axis=None)
        self.assert_array_equal(stat_cube.data, self.cube1.data)

    def test_datetime_to_int_days_no_overlap(self):
        """Test _datetime_to_int_days."""
        computed_dats = _datetime_to_int_days(self.cube1)
        expected_dats = [0, 31]
        self.assert_array_equal(computed_dats, expected_dats)

    def test_assemble_overlap_data(self):
        """Test overlap data."""
        comp_ovlap_mean = _assemble_overlap_data([self.cube1, self.cube1],
                                                 [0, 31], "mean")
        expected_ovlap_mean = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(comp_ovlap_mean.data, expected_ovlap_mean)

    def test_assemble_full_data(self):
        """Test full data."""
        comp_full_mean = _assemble_full_data([self.cube1, self.cube2], "mean")
        expected_full_mean = np.ma.ones((2, 3, 2, 2))
        expected_full_mean.mask = np.zeros((2, 3, 2, 2))
        expected_full_mean.mask[1] = True
        self.assert_array_equal(comp_full_mean.data, expected_full_mean)

    def test_slice_cube(self):
        """Test slice cube."""
        comp_slice = _slice_cube(self.cube1, 0, 31)
        self.assert_array_equal([0, 1], comp_slice)

    def test_get_overlap(self):
        """Test get overlap."""
        full_ovlp = _get_overlap([self.cube1, self.cube1])
        self.assert_array_equal([0, 31], full_ovlp)
        no_ovlp = _get_overlap([self.cube1, self.cube2])
        np.testing.assert_equal(None, no_ovlp)

    def test_plev_fix(self):
        """Test plev fix."""
        fixed_data = _plev_fix(self.cube2.data, 1)
        expected_data = np.ma.ones((3, 2, 2))
        self.assert_array_equal(expected_data, fixed_data)


if __name__ == '__main__':
    unittest.main()
