"""Tests for MIROC5."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg import Siconca
from esmvalcore.cmor.fix import Fix


class TestSiconca(unittest.TestCase):
    """Test sftof fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='siconca', units='%')
        self.fix = Siconca(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(Fix.get_fixes('CMIP6',
                                           'EC-Earth3-Veg',
                                           'SImon', 'siconca'),
                             [Siconca(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('%'))
