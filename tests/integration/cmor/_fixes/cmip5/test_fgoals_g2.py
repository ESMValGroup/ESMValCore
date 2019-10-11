"""Test FGOALS-g2 fixes."""
import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.fgoals_g2 import AllVars


class TestAll(unittest.TestCase):
    """Test fixes for all vars."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='co2', units='J')
        self.cube.add_dim_coord(
            DimCoord(
                [0.0, 1.0],
                standard_name='time',
                units=Unit('days since 0001-01', calendar='gregorian')),
            0)
        self.fix = AllVars()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'FGOALS-G2', 'tas'), [AllVars()])

    def test_fix_metadata(self):
        """Test calendar fix."""
        cube = self.fix.fix_metadata([self.cube])[0]

        time = cube.coord('time')
        self.assertEqual(time.units.origin,
                         'day since 1-01-01 00:00:00.000000')
        self.assertEqual(time.units.calendar, 'gregorian')

    def test_fix_metadata_dont_gail_if_not_time(self):
        """Test calendar fix."""
        self.cube.remove_coord('time')
        self.fix.fix_metadata([self.cube])
