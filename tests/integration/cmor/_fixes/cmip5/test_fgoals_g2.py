"""Test FGOALS-g2 fixes."""
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.fgoals_g2 import AllVars
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cube():
    """Cube for testing."""
    test_cube = Cube([[1.0, 2.0]], var_name='co2', units='J')
    test_cube.add_dim_coord(
        DimCoord(
            [0.0, 1.0],
            standard_name='time',
            units=Unit('days since 0001-01', calendar='gregorian')),
        1)
    test_cube.add_dim_coord(
        DimCoord(
            [180],
            standard_name='longitude',
            units=Unit('degrees')),
        0)
    return test_cube


class TestAll:
    """Test fixes for all vars."""

    @staticmethod
    def test_get():
        """Test fix get."""
        assert (Fix.get_fixes('CMIP5', 'FGOALS-G2', 'Amon', 'tas')
                == [AllVars(None)])

    @staticmethod
    def test_fix_metadata(cube):
        """Test calendar fix."""
        fix = AllVars(None)
        cube = fix.fix_metadata([cube])[0]

        time = cube.coord('time')
        assert time.units.origin == 'day since 1-01-01 00:00:00.000000'
        assert time.units.calendar in ('standard', 'gregorian')

    @staticmethod
    def test_fix_metadata_dont_fail_if_not_longitude(cube):
        """Test calendar fix."""
        cube.remove_coord('longitude')
        fix = AllVars(None)
        fix.fix_metadata([cube])

    @staticmethod
    def test_fix_metadata_dont_fail_if_not_time(cube):
        """Test calendar fix."""
        cube.remove_coord('time')
        fix = AllVars(None)
        fix.fix_metadata([cube])
