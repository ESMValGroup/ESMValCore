"""Unit test for the :func:`esmvalcore.preprocessor._units` function"""

import unittest

import cf_units
import iris
import iris.fileformats
import numpy as np

import tests
from esmvalcore.preprocessor._units import convert_units


class Test(tests.Test):
    """Test class for _units"""

    def setUp(self):
        """Prepare tests"""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        self.data2 = np.array([[0., 1.], [2., 3.]])
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
        coords_spec3 = [(lats2, 0), (lons2, 1)]
        self.arr = iris.cube.Cube(self.data2, units='K',
                                  dim_coords_and_dims=coords_spec3)

    def test_convert_incompatible_units(self):
        """Test conversion to incompatible units."""
        self.assertRaises(ValueError, convert_units, self.arr, 'm')

    def test_convert_compatible_units(self):
        """Test conversion to compatible units."""
        result = convert_units(self.arr, 'degC')
        expected_data = np.array([[-273.15, -272.15], [-271.15, -270.15]])
        expected_units = cf_units.Unit('degC')
        self.assertEqual(result.units, expected_units)
        self.assert_array_equal(result.data, expected_data)

    def test_convert_precipitation_flux(self):
        """Test special conversion of precipitation_flux."""
        self.arr.standard_name = 'precipitation_flux'
        self.arr.units = 'kg m-2 s-1'
        result = convert_units(self.arr, 'mm day-1')
        self.assertEqual(result.standard_name, 'lwe_precipitation_rate')
        self.assertEqual(result.units, 'mm day-1')
        np.testing.assert_allclose(
            result.data,
            [[0.0, 86400.0], [172800.0, 259200.0]],
        )

    def test_convert_precipitation_flux_convertible(self):
        """Test special conversion of precipitation_flux."""
        self.arr.standard_name = 'precipitation_flux'
        self.arr.units = 'g m-2 yr-1'
        result = convert_units(self.arr, 'm yr-1')
        self.assertEqual(result.standard_name, 'lwe_precipitation_rate')
        self.assertEqual(result.units, 'm yr-1')
        np.testing.assert_allclose(
            result.data,
            [[0.0, 1.0e-6], [2.0e-6, 3.0e-6]],
        )

    def test_convert_precipitation_flux_fail_invalid_name(self):
        """Test special conversion of precipitation_flux."""
        self.arr.units = 'kg m-2 s-1'
        self.assertRaises(ValueError, convert_units, self.arr, 'mm day-1')

    def test_convert_precipitation_flux_fail_invalid_source_units(self):
        """Test special conversion of precipitation_flux."""
        self.arr.standard_name = 'precipitation_flux'
        self.assertRaises(ValueError, convert_units, self.arr, 'mm day-1')

    def test_convert_special_pr_fail_invalid_target_units(self):
        """Test special conversion of precipitation_flux."""
        self.arr.standard_name = 'precipitation_flux'
        self.arr.units = 'kg m-2 s-1'
        self.assertRaises(ValueError, convert_units, self.arr, 'K')


if __name__ == '__main__':
    unittest.main()
