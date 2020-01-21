"""Test fixes for ACCESS1-3."""
import unittest

from cf_units import Unit
from iris.cube import Cube
from iris.coords import AuxCoord

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.cmip5.access1_3 import AllVars


class TestAllVars(unittest.TestCase):
    """Test fixes for all vars."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.cube.add_aux_coord(
            AuxCoord(0, 'time', 'time', 'time',
                     Unit('days since 1850-01-01', 'julian')))
        self.fix = AllVars(None)

    def test_get(self):
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'ACCESS1-3', 'Amon', 'tas'),
            [AllVars(None)])

    def test_fix_metadata(self):
        """Test calendar fix."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.coord('time').units.calendar, 'gregorian')

    def test_fix_metadata_if_not_time(self):
        """Test calendar fix do not fail if no time coord present."""
        self.cube.remove_coord('time')
        self.fix.fix_metadata([self.cube])
