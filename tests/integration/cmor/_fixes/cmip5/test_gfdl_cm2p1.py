"""Test GDL-CM2P1 fixes."""
import unittest

from cf_units import Unit
import iris
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.gfdl_cm2p1 import (Sftof, AllVars,
                                                     Areacello, Sit)


class TestSftof(unittest.TestCase):
    """Test sftof fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftof', units='J')
        self.fix = Sftof(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'Amon', 'sftof'),
            [Sftof(None), AllVars(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestAreacello(unittest.TestCase):
    """Test areacello fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='areacello', units='m-2')
        self.fix = Areacello(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'Amon', 'areacello'),
            [Areacello(None), AllVars(None)])

    def test_fix_metadata(self):
        """Test data fix."""
        cube = self.fix.fix_metadata((self.cube, ))[0]
        self.assertEqual(cube.data[0], 1.0)
        self.assertEqual(cube.units, Unit('m2'))

    def test_fix_data(self):
        """Test data fix."""
        self.cube.units = 'm2'
        cube = self.fix.fix_metadata((self.cube, ))[0]
        self.assertEqual(cube.data[0], 1.0)
        self.assertEqual(cube.units, Unit('m2'))


class TestSit(unittest.TestCase):
    """Test sit fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='sit', units='m')
        self.cube.add_dim_coord(
            iris.coords.DimCoord(
                points=[45000.5, 45001.5],
                var_name='time',
                standard_name='time',
                long_name='time',
                units='days since 1850-01-01',
                bounds=[[1e8, 1.1e8], [1.1e8, 1.2e8]]
            ),
            0
        )
        self.fix = Sit(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'OImon', 'sit'),
            [Sit(None), AllVars(None)])

    def test_fix_metadata(self):
        """Test data fix."""
        cube = self.fix.fix_metadata((self.cube,))[0]
        time = cube.coord('time')
        self.assertEqual(time.bounds[0, 0], 45000)
        self.assertEqual(time.bounds[0, 1], 45001)
        self.assertEqual(time.bounds[1, 0], 45001)
        self.assertEqual(time.bounds[1, 1], 45002)
