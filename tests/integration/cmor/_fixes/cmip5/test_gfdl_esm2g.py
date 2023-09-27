"""Tests for fixes of GFDL-ESM2G (CMIP5)."""
import unittest
from unittest import mock

import iris
import pytest
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.gfdl_esm2g import (
    AllVars,
    Areacello,
    Co2,
    FgCo2,
    Usi,
    Vsi,
    _get_and_remove,
)
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info

CUBE_1 = iris.cube.Cube([1.0], long_name='to_be_rm')
CUBE_2 = iris.cube.Cube([1.0], long_name='not_to_be_rm')
CUBES_LISTS = [
    (iris.cube.CubeList([CUBE_1]), iris.cube.CubeList([])),
    (iris.cube.CubeList([CUBE_1, CUBE_2]), iris.cube.CubeList([CUBE_2])),
    (iris.cube.CubeList([CUBE_2]), iris.cube.CubeList([CUBE_2])),
]


@pytest.mark.parametrize('cubes_in,cubes_out', CUBES_LISTS)
def test_get_and_remove(cubes_in, cubes_out):
    """Test get and remove helper method."""
    _get_and_remove(cubes_in, 'to_be_rm')
    assert cubes_in is not cubes_out
    assert cubes_in == cubes_out


CUBES = iris.cube.CubeList([CUBE_1, CUBE_2])


@mock.patch('esmvalcore.cmor._fixes.cmip5.gfdl_esm2g._get_and_remove',
            autospec=True)
def test_allvars(mock_get_and_remove):
    """Test fixes for all vars."""
    fix = AllVars(None)
    fix.fix_metadata(CUBES)
    assert mock_get_and_remove.call_count == 3
    assert mock_get_and_remove.call_args_list == [
        mock.call(CUBES, 'Start time for average period'),
        mock.call(CUBES, 'End time for average period'),
        mock.call(CUBES, 'Length of average period'),
    ]


@mock.patch('esmvalcore.cmor._fixes.cmip5.gfdl_esm2g._get_and_remove',
            autospec=True)
def test_fgco2(mock_get_and_remove):
    """Test fgco2 fixes."""
    fix = FgCo2(None)
    fix.fix_metadata(CUBES)
    assert mock_get_and_remove.call_count == 2
    assert mock_get_and_remove.call_args_list == [
        mock.call(CUBES, 'Latitude of tracer (h) points'),
        mock.call(CUBES, 'Longitude of tracer (h) points'),
    ]


class TestCo2(unittest.TestCase):
    """Test co2 fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = iris.cube.Cube([1.0], var_name='co2', units='J')
        self.fix = Co2(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2G', 'Amon', 'co2'),
            [Co2(None), AllVars(None), GenericFix(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1e6)
        self.assertEqual(cube.units, Unit('J'))


class TestUsi(unittest.TestCase):
    """Test usi fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = iris.cube.Cube([1.0], var_name='usi', units='J')
        self.vardef = get_var_info('CMIP5', 'day', self.cube.var_name)
        self.fix = Usi(self.vardef)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2G', 'day', 'usi'),
            [Usi(self.vardef),
             AllVars(self.vardef),
             GenericFix(self.vardef)])

    def test_fix_data(self):
        """Test metadata fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.standard_name, 'sea_ice_x_velocity')


class TestVsi(unittest.TestCase):
    """Test vsi fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = iris.cube.Cube([1.0], var_name='vsi', units='J')
        self.vardef = get_var_info('CMIP5', 'day', self.cube.var_name)
        self.fix = Vsi(self.vardef)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'GFDL-ESM2G', 'day', 'vsi'),
            [Vsi(self.vardef),
             AllVars(self.vardef),
             GenericFix(self.vardef)])

    def test_fix_data(self):
        """Test metadata fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.standard_name, 'sea_ice_y_velocity')


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
            Fix.get_fixes('CMIP5', 'GFDL-ESM2G', 'fx', 'areacello'),
            [Areacello(self.vardef),
             AllVars(self.vardef),
             GenericFix(self.vardef)])

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
