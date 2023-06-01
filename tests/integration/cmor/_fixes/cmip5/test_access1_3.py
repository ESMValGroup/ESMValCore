"""Test fixes for ACCESS1-3."""
from datetime import datetime

import pytest
from cf_units import Unit, num2date
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.access1_0 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip5.access1_3 import AllVars, Cl
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.iris_helpers import date2num


@pytest.fixture
def cube():
    """Cube for testing."""
    test_cube = Cube([1.0, 2.0], var_name='co2', units='J')
    reference_dates = [
        datetime(300, 1, 16, 12),  # e.g. piControl
        datetime(1850, 1, 16, 12)  # e.g. historical
    ]
    esgf_time_units = Unit(
        'days since 0001-01-01',
        calendar='proleptic_gregorian',
    )
    time_points = date2num(reference_dates, esgf_time_units)
    test_cube.add_dim_coord(
        DimCoord(time_points, 'time', 'time', 'time', esgf_time_units),
        data_dim=0,
    )
    return test_cube


class TestAllVars:
    """Test fixes for all vars."""

    @staticmethod
    def test_get():
        """Test getting of fix."""
        assert (Fix.get_fixes('CMIP5', 'ACCESS1-3', 'Amon', 'tas')
                == [AllVars(None)])

    @staticmethod
    def test_fix_metadata(cube):
        """Test fix for bad calendar."""
        fix = AllVars(None)
        cube = fix.fix_metadata([cube])[0]
        time = cube.coord('time')
        dates = num2date(time.points, time.units.name, time.units.calendar)
        assert time.units.calendar in ('standard', 'gregorian')
        u = Unit('days since 300-01-01 12:00:00', calendar='standard')
        assert dates[0] == u.num2date(15)
        u = Unit('days since 1850-01-01 12:00:00', calendar='standard')
        assert dates[1] == u.num2date(15)

    @staticmethod
    def test_fix_metadata_if_not_time(cube):
        """Test calendar fix do not fail if no time coord present."""
        cube.remove_coord('time')
        fix = AllVars(None)
        fix.fix_metadata([cube])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'ACCESS1-3', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl
