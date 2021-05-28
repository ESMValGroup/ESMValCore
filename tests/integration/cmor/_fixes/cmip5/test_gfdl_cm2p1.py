"""Test GDL-CM2P1 fixes."""
import unittest
from unittest import mock

import iris
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.gfdl_cm2p1 import (AllVars, Areacello, Cl,
                                                     Sftof, Sit)
from esmvalcore.cmor._fixes.cmip5.cesm1_cam5 import Cl as BaseCl
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestCl(unittest.TestCase):
    """Test cl fix."""
    def test_get(self):
        """Test getting of fix."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'Amon', 'cl'),
            [Cl(None), AllVars(None)])

    def test_fix(self):
        """Test fix for ``cl``."""
        assert Cl is BaseCl


class TestSftof(unittest.TestCase):
    """Test sftof fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftof', units='J')
        self.fix = Sftof(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'fx', 'sftof'),
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
        self.vardef = get_var_info('CMIP5', 'fx', self.cube.var_name)
        self.fix = Areacello(self.vardef)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'Amon', 'areacello'),
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


class TestSit(unittest.TestCase):
    """Test sit fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='sit', units='m')
        self.cube.add_dim_coord(
            iris.coords.DimCoord(
                points=[45000.5, 45031.5],
                var_name='time',
                standard_name='time',
                long_name='time',
                units='days since 1850-01-01',
                bounds=[[1e8, 1.1e8], [1.1e8, 1.2e8]]
            ),
            0
        )
        self.var_info_mock = mock.Mock()
        self.var_info_mock.frequency = 'mon'
        self.var_info_mock.short_name = 'sit'
        self.fix = Sit(self.var_info_mock)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-CM2P1', 'OImon', 'sit'),
            [Sit(self.var_info_mock), AllVars(None)])

    def test_fix_metadata_day_do_nothing(self):
        """Test data fix."""
        self.var_info_mock.frequency = 'day'
        fix = Sit(self.var_info_mock)
        cube = fix.fix_metadata((self.cube,))[0]
        time = cube.coord('time')
        self.assertEqual(time.bounds[0, 0], 1e8)
        self.assertEqual(time.bounds[0, 1], 1.1e8)
        self.assertEqual(time.bounds[1, 0], 1.1e8)
        self.assertEqual(time.bounds[1, 1], 1.2e8)

    def test_fix_metadata(self):
        """Test data fix."""
        fix = Sit(self.var_info_mock)
        cube = fix.fix_metadata((self.cube,))[0]
        time = cube.coord('time')
        self.assertEqual(time.bounds[0, 0], 44984)
        self.assertEqual(time.bounds[0, 1], 45015)
        self.assertEqual(time.bounds[1, 0], 45015)
        self.assertEqual(time.bounds[1, 1], 45045)

    def test_fix_metadata_not_needed(self):
        """Test data fix."""
        fix = Sit(self.var_info_mock)
        cube = fix.fix_metadata((self.cube,))[0]
        time = cube.coord('time')
        new_bounds = [[44985., 45014.], [45016., 45044.]]
        time.bounds = new_bounds
        self.assertEqual(time.bounds[0, 0], 44985)
        self.assertEqual(time.bounds[0, 1], 45014)
        self.assertEqual(time.bounds[1, 0], 45016)
        self.assertEqual(time.bounds[1, 1], 45044)
