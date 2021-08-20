"""Unit test for the :func:`esmvalcore.preprocessor._units` function."""

import unittest

import cf_units
import iris
import numpy as np

import tests
from esmvalcore.preprocessor._units import convert_units, flux_to_total


class TestConvertUnits(tests.Test):
    """Test class for _units."""
    def setUp(self):
        """Prepare tests."""
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
        self.arr = iris.cube.Cube(self.data2,
                                  units='K',
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


class TestFluxToTotal(tests.Test):
    """Test class for _units."""
    def setUp(self):
        """Prepare tests."""
        data = np.arange(4)
        time = iris.coords.DimCoord(
            np.arange(1, 8, 2),
            standard_name='time',
            bounds=np.array([np.arange(0, 8, 2),
                             np.arange(2, 9, 2)]).T,
            units='days since 1850-01-01')

        coords_spec = [
            (time, 0),
        ]
        self.cube = iris.cube.Cube(data,
                                   units='kg day-1',
                                   dim_coords_and_dims=coords_spec)

    def test_convert_incompatible_units(self):
        """Test conversion to incompatible units."""
        self.cube.units = 'kg m-2'
        self.assertRaises(ValueError, flux_to_total, self.cube)

    def test_flux_by_second(self):
        """Test conversion to compatible units."""
        self.cube.units = 'kg s-1'
        result = flux_to_total(self.cube)
        expected_data = np.array([0, 2, 4, 6]) * 24 * 3600
        expected_units = cf_units.Unit('kg')
        self.assertEqual(result.units, expected_units)
        self.assert_array_equal(result.data, expected_data)

    def test_flux_by_day(self):
        """Test conversion to compatible units."""
        result = flux_to_total(self.cube)
        expected_data = np.array([0, 2, 4, 6])
        expected_units = cf_units.Unit('kg')
        self.assertEqual(result.units, expected_units)
        self.assert_array_equal(result.data, expected_data)

    def test_flux_by_hour(self):
        """Test conversion to compatible units."""
        self.cube.units = 'kg hr-1'
        result = flux_to_total(self.cube)
        expected_data = np.array([0, 2, 4, 6]) * 24
        expected_units = cf_units.Unit('kg')
        self.assertEqual(result.units, expected_units)
        self.assert_array_equal(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
