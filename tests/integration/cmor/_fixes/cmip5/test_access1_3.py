"""Test fixes for ACCESS1-3."""
import unittest

from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.access1_0 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.access1_3 import AllVars, Cl
from esmvalcore.cmor._fixes.fix import Fix


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
        """Test getting of fix."""
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


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'ACCESS1-3', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl
