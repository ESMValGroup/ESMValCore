"""Test for the fixes for EC-Earth3-HR model from PRIMAVERA project."""
import unittest

from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.primavera.ec_earth3_hr import AllVars


class TestAllVars(unittest.TestCase):
    """Test for the common fixes for all the variables."""

    def setUp(self):
        """Prepare to test."""
        self.cube = Cube([[1.0, 2.0], [3.0, 4.0]], var_name='var')
        self.cube.add_dim_coord(
            DimCoord([1.0, 2.0], standard_name='latitude',
                     var_name='latitude'), 0)
        self.cube.add_dim_coord(
            DimCoord([1.0, 2.0], standard_name='longitude',
                     var_name='longitude'), 1)
        self.fix = AllVars()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('PRIMAVERA', 'EC-EARTH3-HR', 'tos'), [AllVars()]
        )

    def test_fix_lat_lon_names(self):
        """
        Test latitude and longitude var names.

        They should be lat and lon instead of the original latitude and
        longitude
        """
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.coord('latitude').var_name, 'lat')
        self.assertEqual(cube.coord('longitude').var_name, 'lon')
