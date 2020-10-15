"""Test FGOALS-g2 fixes."""
import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip5.fgoals_g2 import AllVars, Cl
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestAll(unittest.TestCase):
    """Test fixes for all vars."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([[1.0, 2.0]], var_name='co2', units='J')
        self.cube.add_dim_coord(
            DimCoord(
                [0.0, 1.0],
                standard_name='time',
                units=Unit('days since 0001-01', calendar='gregorian')),
            1)
        self.cube.add_dim_coord(
            DimCoord(
                [180],
                standard_name='longitude',
                units=Unit('degrees')),
            0)
        self.fix = AllVars(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'FGOALS-G2', 'Amon', 'tas'),
            [AllVars(None)])

    def test_fix_metadata(self):
        """Test calendar fix."""
        cube = self.fix.fix_metadata([self.cube])[0]

        time = cube.coord('time')
        self.assertEqual(time.units.origin,
                         'day since 1-01-01 00:00:00.000000')
        self.assertEqual(time.units.calendar, 'gregorian')

    def test_fix_metadata_dont_fail_if_not_longitude(self):
        """Test calendar fix."""
        self.cube.remove_coord('longitude')
        self.fix.fix_metadata([self.cube])

    def test_fix_metadata_dont_fail_if_not_time(self):
        """Test calendar fix."""
        self.cube.remove_coord('time')
        self.fix.fix_metadata([self.cube])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'FGOALS-g2', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.fgoals_g2.add_sigma_factory',
    autospec=True)
def test_cl_fix_metadata(mock_add_sigma_factory):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = CubeList([Cube(0.0, var_name='cl'), Cube(1.0, var_name='x')])
    vardef = get_var_info('CMIP6', 'Amon', 'cl')
    fix = Cl(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    mock_add_sigma_factory.assert_called_once_with(cubes[0])
    assert fixed_cubes == CubeList([cubes[0]])
