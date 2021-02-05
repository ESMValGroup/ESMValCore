"""Unit test for :func:`esmvalcore.preprocessor._multimodel`."""

import unittest

import iris
import numpy as np
from cf_units import Unit

import tests
from esmvalcore.preprocessor import multi_model_statistics
from esmvalcore.preprocessor._multimodel import (
    _assemble_data,
    _compute_statistic,
    _get_time_slice,
    _multiproduct_statistics,
    _plev_fix,
    _put_in_cube,
    _unify_time_coordinates,
)


class Test(tests.Test):
    """Test class for preprocessor/_multimodel.py."""
    def setUp(self):
        """Prepare tests."""
        # Make various time arrays
        time_args = {
            'standard_name': 'time',
            'units': Unit('days since 1850-01-01', calendar='gregorian')
        }
        monthly1 = iris.coords.DimCoord([14, 45], **time_args)
        monthly2 = iris.coords.DimCoord([45, 73, 104, 134], **time_args)
        monthly3 = iris.coords.DimCoord([104, 134], **time_args)
        yearly1 = iris.coords.DimCoord([14., 410.], **time_args)
        yearly2 = iris.coords.DimCoord([1., 367., 733., 1099.], **time_args)
        daily1 = iris.coords.DimCoord([1., 2.], **time_args)
        for time in [monthly1, monthly2, monthly3, yearly1, yearly2, daily1]:
            time.guess_bounds()

        # Other dimensions are fixed
        zcoord = iris.coords.DimCoord([0.5, 5., 50.],
                                      standard_name='air_pressure',
                                      long_name='air_pressure',
                                      bounds=[[0., 2.5], [2.5, 25.],
                                              [25., 250.]],
                                      units='m',
                                      attributes={'positive': 'down'})
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
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

        data1 = np.ma.ones((2, 3, 2, 2))
        data2 = np.ma.ones((4, 3, 2, 2))
        mask2 = np.full((4, 3, 2, 2), False)
        mask2[0, 0, 0, 0] = True
        data2 = np.ma.array(data2, mask=mask2)

        coords_spec1 = [(monthly1, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube1 = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec1)

        coords_spec2 = [(monthly2, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube2 = iris.cube.Cube(data2, dim_coords_and_dims=coords_spec2)

        coords_spec3 = [(monthly3, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube3 = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec3)

        coords_spec4 = [(yearly1, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube4 = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec4)

        coords_spec5 = [(yearly2, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube5 = iris.cube.Cube(data2, dim_coords_and_dims=coords_spec5)

        coords_spec6 = [(daily1, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube6 = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec6)

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
        stats = multi_model_statistics(products=data,
                                       statistics=['mean'],
                                       span='full')
        expected_full_mean = np.ma.ones((5, 3, 2, 2))
        expected_full_mean.mask = np.ones((5, 3, 2, 2))
        expected_full_mean.mask[1] = False
        self.assert_array_equal(stats['mean'].data, expected_full_mean)

    def test_compute_full_statistic_yr_cube(self):
        data = [self.cube4, self.cube5]
        stats = multi_model_statistics(products=data,
                                       statistics=['mean'],
                                       span='full')
        expected_full_mean = np.ma.ones((4, 3, 2, 2))
        expected_full_mean.mask = np.zeros((4, 3, 2, 2))
        expected_full_mean.mask[2:4] = True
        self.assert_array_equal(stats['mean'].data, expected_full_mean)

    def test_compute_overlap_statistic_mon_cube(self):
        data = [self.cube1, self.cube1]
        stats = multi_model_statistics(products=data,
                                       statistics=['mean'],
                                       span='overlap')
        expected_ovlap_mean = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(stats['mean'].data, expected_ovlap_mean)

    def test_compute_overlap_statistic_yr_cube(self):
        data = [self.cube4, self.cube4]
        stats = multi_model_statistics(products=data,
                                       statistics=['mean'],
                                       span='overlap')
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
        stat_cube = _put_in_cube(self.cube1, cube_data, "mean", t_axis=[1, 2])
        self.assert_array_equal(stat_cube.data, self.cube1.data)

    def test_assemble_overlap_data(self):
        """Test overlap data."""
        comp_ovlap_mean = _assemble_data([self.cube1, self.cube1],
                                         "mean",
                                         span='overlap')
        expected_ovlap_mean = np.ma.ones((2, 3, 2, 2))
        self.assert_array_equal(comp_ovlap_mean.data, expected_ovlap_mean)

    def test_assemble_full_data(self):
        """Test full data."""
        comp_full_mean = _assemble_data([self.cube1, self.cube2],
                                        "mean",
                                        span='full')
        expected_full_mean = np.ma.ones((5, 3, 2, 2))
        expected_full_mean.mask = np.ones((5, 3, 2, 2))
        expected_full_mean.mask[1] = False
        self.assert_array_equal(comp_full_mean.data, expected_full_mean)

    def test_plev_fix(self):
        """Test plev fix."""
        fixed_data = _plev_fix(self.cube2.data, 1)
        expected_data = np.ma.ones((3, 2, 2))
        self.assert_array_equal(expected_data, fixed_data)

    def test_unify_time_coordinates(self):
        """Test set common calenar."""
        cube1 = self.cube1
        time1 = cube1.coord('time')
        t_unit1 = time1.units
        dates = t_unit1.num2date(time1.points)

        t_unit2 = Unit('days since 1850-01-01', calendar='gregorian')
        time2 = t_unit2.date2num(dates)
        cube2 = self.cube1.copy()
        cube2.coord('time').points = time2
        cube2.coord('time').units = t_unit2
        _unify_time_coordinates([cube1, cube2])
        self.assertEqual(cube1.coord('time'), cube2.coord('time'))

    def test_get_time_slice_all(self):
        """Test get time slice if all cubes have data."""
        cubes = [self.cube1, self.cube2]
        result = _get_time_slice(cubes, time=45)
        expected = [self.cube1[1].data, self.cube2[0].data]
        self.assert_array_equal(expected, result)

    def test_get_time_slice_part(self):
        """Test get time slice if all cubes have data."""
        cubes = [self.cube1, self.cube2]
        result = _get_time_slice(cubes, time=14)
        masked = np.ma.empty(list(cubes[0].shape[1:]))
        masked.mask = True
        expected = [self.cube1[0].data, masked]
        self.assert_array_equal(expected, result)

    def test_return_products(self):
        """Check that the right product set is returned."""
        input1 = PreprocessorFile(self.cube1)
        input2 = PreprocessorFile(self.cube2)
        output = PreprocessorFile()
        products = set([input1, input2])
        output_products = {'mean': output}
        result1 = _multiproduct_statistics(products,
                                           statistics=['mean'],
                                           output_products=output_products,
                                           span='full',
                                           keep_input_datasets=True)
        result2 = _multiproduct_statistics(products,
                                           statistics=['mean'],
                                           output_products=output_products,
                                           span='full',
                                           keep_input_datasets=False)
        result3 = multi_model_statistics(products,
                                         statistics=['mean'],
                                         output_products=output_products,
                                         span='full')
        result4 = multi_model_statistics(products,
                                         statistics=['mean'],
                                         output_products=output_products,
                                         span='full',
                                         keep_input_datasets=False)
        assert result1 == set([input1, input2, output])
        assert result2 == set([output])
        assert result3 == result1
        assert result4 == result2


class PreprocessorFile:
    """Mockup to test output of multimodel."""
    def __init__(self, cube=None):
        if cube:
            self.cubes = [cube]

    def wasderivedfrom(self, product):
        pass


if __name__ == '__main__':
    unittest.main()
