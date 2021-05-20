"""Test CNRM-CM5 fixes."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.cnrm_cm5 import Msftmyz, Msftmyzba


class TestMsftmyz(unittest.TestCase):
    """Test msftmyz fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='msftmyz', units='J')
        self.fix = Msftmyz(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CNRM-CM5', 'Amon', 'msftmyz'),
            [Msftmyz(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1.0e6)
        self.assertEqual(cube.units, Unit('J'))


class TestMsftmyzba(unittest.TestCase):
    """Test msftmyzba fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='msftmyzba', units='J')
        self.fix = Msftmyzba(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CNRM-CM5', 'Amon', 'msftmyzba'),
            [Msftmyzba(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1.0e6)
        self.assertEqual(cube.units, Unit('J'))
