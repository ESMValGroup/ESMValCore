"""Tests for CESM1-BGC fixes."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.cesm1_bgc import Co2


class TestCo2(unittest.TestCase):
    """Tests for co2."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CESM1-BGC', 'co2'), [Co2()])

    def test_fix_data(self):
        """Test fix to set units correctly."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 28.966 / 44.0)
        self.assertEqual(cube.units, Unit('J'))
