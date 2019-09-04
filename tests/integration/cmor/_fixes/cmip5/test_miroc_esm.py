"""Test MIROC-ESM fixes."""
import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.miroc_esm import AllVars, Co2, Tro3


class TestCo2(unittest.TestCase):
    """Test c02 fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'co2'), [AllVars(), Co2()])

    def test_fix_metadata(self):
        """Test unit fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.data[0], 1)
        self.assertEqual(cube.units, Unit('1e-6'))


class TestTro3(unittest.TestCase):
    """Test tro3 fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='tro3', units='J')
        self.fix = Tro3()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'tro3'), [AllVars(), Tro3()])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1000)
        self.assertEqual(cube.units, Unit('J'))


class TestAll(unittest.TestCase):
    """Test fixes for allvars."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([[1.0, 2.0], [3.0, 4.0]], var_name='co2', units='J')
        self.cube.add_dim_coord(
            DimCoord(
                [0, 1],
                standard_name='time',
                units=Unit(
                    'days since 0000-01-01 00:00:00', calendar='gregorian')),
            0)
        self.cube.add_dim_coord(DimCoord([0, 1], long_name='AR5PL35'), 1)

        self.fix = AllVars()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'tos'), [AllVars()])

    def test_fix_metadata_plev(self):
        """Test plev fix."""
        time = self.cube.coord('time')
        time.units = Unit("days since 1-1-1", time.units.calendar)
        cube = self.fix.fix_metadata([self.cube])[0]
        cube.coord('air_pressure')

    def test_fix_metadata_no_plev(self):
        """Test plev fix wotk with no plev."""
        self.cube.remove_coord('AR5PL35')
        cube = self.fix.fix_metadata([self.cube])[0]
        with self.assertRaises(CoordinateNotFoundError):
            cube.coord('air_pressure')
