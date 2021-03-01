"""Test fixes for FIO-ESM."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.fio_esm import Ch4, Cl, Co2
from esmvalcore.cmor._fixes.cmip5.cesm1_cam5 import Cl as BaseCl


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'FIO-ESM', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


class TestCh4(unittest.TestCase):
    """Test ch4 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='ch4', units='J')
        self.fix = Ch4(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'FIO-ESM', 'Amon', 'ch4'),
                             [Ch4(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 29. / 16. * 1.e9)
        self.assertEqual(cube.units, Unit('J'))


class TestCo2(unittest.TestCase):
    """Test co2 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'FIO-ESM', 'Amon', 'co2'),
                             [Co2(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 29. / 44. * 1.e6)
        self.assertEqual(cube.units, Unit('J'))
