"""Test CANESM2 fixes."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.canesm2 import FgCo2
from esmvalcore.cmor.fix import Fix


class TestCanESM2Fgco2(unittest.TestCase):
    """Test fgc02 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='fgco2', units='J')
        self.fix = FgCo2(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CANESM2', 'Amon', 'fgco2'), [FgCo2(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 12.0 / 44.0)
        self.assertEqual(cube.units, Unit('J'))
