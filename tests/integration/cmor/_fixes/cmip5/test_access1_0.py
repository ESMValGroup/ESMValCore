"""Test Access1-0 fixes."""
import unittest

from cf_units import Unit
from iris.cube import Cube
from iris.coords import AuxCoord

from esmvalcore.cmor._fixes.cmip5.access1_0 import AllVars


class TestAllVars(unittest.TestCase):
    """Test all vars fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.cube.add_aux_coord(AuxCoord(
            0,
            'time',
            'time',
            'time',
            Unit('days since 1850-01-01', 'julian')
        ))
        self.fix = AllVars()

    def test_fix_metadata(self):
        """Test fix for bad calendar."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.coord('time').units.calendar, 'gregorian')

    def test_fix_metadata_if_not_time(self):
        """Test calendar fix do not fail if no time coord present."""
        self.cube.remove_coord('time')
        self.fix.fix_metadata([self.cube])
