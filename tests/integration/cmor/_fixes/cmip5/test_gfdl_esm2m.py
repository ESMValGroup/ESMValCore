"""Test fixes for GFDL-ES2M."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.gfdl_esm2m import (AllVars, Areacello, Co2,
                                                     Sftof)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestSftof(unittest.TestCase):
    """Test sftof fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftof', units='J')
        self.fix = Sftof(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2M', 'fx', 'sftof'),
            [Sftof(None), AllVars(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestCo2(unittest.TestCase):
    """Test co2 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.fix = Co2(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2M', 'Amon', 'co2'),
            [Co2(None), AllVars(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1e6)
        self.assertEqual(cube.units, Unit('J'))


class TestAreacello(unittest.TestCase):
    """Test sftof fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='areacello', units='m-2')
        self.vardef = get_var_info('CMIP5', 'fx', self.cube.var_name)
        self.fix = Areacello(self.vardef)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2M', 'fx', 'areacello'),
            [Areacello(self.vardef), AllVars(self.vardef)])

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
