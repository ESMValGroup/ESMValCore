"""Test MIROC-ESM fixes."""
import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.cmor._fixes.cmip5.miroc_esm import AllVars, Cl, Co2, Tro3
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MIROC-ESM', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


class TestCo2(unittest.TestCase):
    """Test c02 fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.vardef = get_var_info('CMIP5', 'Amon', self.cube.var_name)
        self.fix = Co2(self.vardef)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'Amon', 'co2'),
            [Co2(self.vardef), AllVars(self.vardef)])

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
        self.fix = Tro3(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'Amon', 'tro3'),
            [Tro3(None), AllVars(None)])

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
            DimCoord([0, 1],
                     standard_name='time',
                     units=Unit('days since 0000-01-01 00:00:00',
                                calendar='gregorian')), 0)
        self.cube.add_dim_coord(DimCoord([0, 1], long_name='AR5PL35'), 1)

        time_units = Unit('days since 1950-1-1 00:00:00', calendar='gregorian')

        # Setup wrong time coordinate that is present in some files
        # (-711860.5 days from 1950-01-01 is < year 1)
        time_coord = DimCoord(
            [-711845.0, -711814.0],
            bounds=[[-711860.5, -711829.5], [-711829.5, -711800.0]],
            var_name='time',
            standard_name='time',
            long_name='time',
            units=time_units,
        )
        self.cube_with_wrong_time = Cube([0.0, 1.0], var_name='co2',
                                         units='ppm',
                                         dim_coords_and_dims=[(time_coord, 0)])

        self.fix = AllVars(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MIROC-ESM', 'Amon', 'tos'),
            [AllVars(None)])

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

    def test_fix_metadata_correct_time(self):
        """Test fix for time."""
        fixed_cube = self.fix.fix_metadata([self.cube])[0]
        time_coord = fixed_cube.coord('time')
        np.testing.assert_allclose(time_coord.points, [0, 1])
        assert time_coord.bounds is None

    def test_fix_metadata_wrong_time(self):
        """Test fix for time."""
        fixed_cube = self.fix.fix_metadata([self.cube_with_wrong_time])[0]
        time_coord = fixed_cube.coord('time')
        np.testing.assert_allclose(time_coord.points, [-711841.5, -711810.5])
        np.testing.assert_allclose(
            time_coord.bounds,
            [[-711857.0, -711826.0], [-711826.0, -711796.5]])

    def test_fix_metadata_wrong_time_no_bounds(self):
        """Test fix for time."""
        self.cube_with_wrong_time.coord('time').bounds = None
        fixed_cube = self.fix.fix_metadata([self.cube_with_wrong_time])[0]
        time_coord = fixed_cube.coord('time')
        np.testing.assert_allclose(time_coord.points, [-711845.0, -711814.0])
        assert time_coord.bounds is None
