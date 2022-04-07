"""Test fixes for ACCESS1-3."""
import unittest
from datetime import datetime

from cf_units import Unit, num2date
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.access1_0 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.access1_3 import AllVars, Cl
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.iris_helpers import date2num


class TestAllVars(unittest.TestCase):
    """Test fixes for all vars."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0, 2.0], var_name='co2', units='J')
        reference_dates = [
            datetime(300, 1, 16, 12),  # e.g. piControl
            datetime(1850, 1, 16, 12)  # e.g. historical
        ]
        esgf_time_units = Unit('days since 0001-01-01',
                               calendar='proleptic_gregorian')
        time_points = date2num(reference_dates, esgf_time_units)
        self.cube.add_dim_coord(
            DimCoord(time_points, 'time', 'time', 'time', esgf_time_units),
            data_dim=0)
        self.fix = AllVars(None)

    def test_get(self):
        """Test getting of fix."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'ACCESS1-3', 'Amon', 'tas'),
            [AllVars(None)])

    def test_fix_metadata(self):
        """Test fix for bad calendar."""
        cube = self.fix.fix_metadata([self.cube])[0]
        time = cube.coord('time')
        dates = num2date(time.points, time.units.name, time.units.calendar)
        self.assertEqual(time.units.calendar, 'gregorian')
        u = Unit('days since 300-01-01 12:00:00', calendar='gregorian')
        self.assertEqual(dates[0], u.num2date(15))
        u = Unit('days since 1850-01-01 12:00:00', calendar='gregorian')
        self.assertEqual(dates[1], u.num2date(15))

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
