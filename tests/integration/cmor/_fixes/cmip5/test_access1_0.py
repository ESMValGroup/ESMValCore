"""Test Access1-0 fixes."""
import unittest

import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip5.access1_0 import AllVars, Cl
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestAllVars(unittest.TestCase):
    """Test all vars fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='co2', units='J')
        self.cube.add_aux_coord(
            AuxCoord(0, 'time', 'time', 'time',
                     Unit('days since 0001-01-01', 'proleptic_gregorian')))
        self.fix = AllVars(None)

    def test_get(self):
        """Test getting of fix."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'ACCESS1-0', 'Amon', 'tas'),
            [AllVars(None)])

    def test_fix_metadata(self):
        """Test fix for bad calendar."""
        cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(cube.coord('time').units.calendar, 'gregorian')

    def test_fix_metadata_if_not_time(self):
        """Test calendar fix do not fail if no time coord present."""
        self.cube.remove_coord('time')
        self.fix.fix_metadata([self.cube])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'ACCESS1-0', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None)]


@pytest.fixture
def cl_cubes():
    """Cubes for ``cl.``."""
    b_coord = AuxCoord(
        [1.0],
        var_name='b',
        long_name='vertical coordinate formula term: b(k)',
        attributes={'a': 1, 'b': '2'},
    )
    cl_cube = Cube(
        [0.0],
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        aux_coords_and_dims=[(b_coord.copy(), 0)],
    )
    x_cube = Cube([0.0],
                  long_name='x',
                  aux_coords_and_dims=[(b_coord.copy(), 0)])
    cubes = CubeList([cl_cube, x_cube])
    return cubes


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.access1_0.ClFixHybridHeightCoord.'
    'fix_metadata', autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata, cl_cubes):
    """Test ``fix_metadata`` for ``cl``."""
    mock_base_fix_metadata.return_value = cl_cubes
    fix = Cl(get_var_info('CMIP5', 'Amon', 'cl'))
    fixed_cubes = fix.fix_metadata(cl_cubes)
    mock_base_fix_metadata.assert_called_once_with(fix, cl_cubes)
    assert len(fixed_cubes) == 2
    cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    b_coord_cl = cl_cube.coord('vertical coordinate formula term: b(k)')
    assert not b_coord_cl.attributes
    x_cube = fixed_cubes.extract_strict('x')
    b_coord_x = x_cube.coord('vertical coordinate formula term: b(k)')
    assert b_coord_x.attributes == {'a': 1, 'b': '2'}
