"""Test MIROC-ESM-CHEM fixes."""

import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.miroc_esm_chem import Tro3


class TestTro3(unittest.TestCase):
    """Test tro3 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='tro3', units='J')
        self.fix = Tro3(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM-CHEM', 'Amon', 'tro3'),
            [Tro3(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1000)
        self.assertEqual(cube.units, Unit('J'))
