"""Test fixes for BNU-ESM."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.bnu_esm import Ch4, Co2, FgCo2, SpCo2


class TestCo2(unittest.TestCase):
    """Test fixes for CO2."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2()

    def test_fix_metadata(self):
        """Test unit change."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.units, Unit('1e-6'))
        self.assertEqual(cube.data[0], 1.0)

    def test_fix_data(self):
        """Test fix values."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 29.0 / 44.0 * 1.e6)
        self.assertEqual(cube.units, Unit('J'))


class Testfgco2(unittest.TestCase):
    """Test fixes for FgCO2."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='fgco2', units='J')
        self.fix = FgCo2()

    def test_fix_metadata(self):
        """Test unit fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.units, Unit('kg m-2 s-1'))
        self.assertEqual(cube.data[0], 1)

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 12.0 / 44.0)
        self.assertEqual(cube.units, Unit('J'))


class TestCh4(unittest.TestCase):
    """Test fixes for ch4."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='ch4', units='J')
        self.fix = Ch4()

    def test_fix_metadata(self):
        """Test unit fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.units, Unit('1e-9'))
        self.assertEqual(cube.data[0], 1)

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 29.0 / 16.0 * 1.e9)
        self.assertEqual(cube.units, Unit('J'))


class Testspco2(unittest.TestCase):
    """Test fixes for SpCO2."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='spco2', units='J')
        self.fix = SpCo2()

    def test_fix_metadata(self):
        """Test fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.units, Unit('J'))
        self.assertEqual(cube.data[0], 1)

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1.e6)
        self.assertEqual(cube.units, Unit('J'))
