"""Tests for MIROC5."""
import unittest

import iris
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.miroc5 import Cl, Hur, Pr, Sftof, Tas
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MIROC5', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_hur_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MIROC5', 'Amon', 'hur')
    assert fix == [Hur(None)]


def test_get_pr_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MIROC5', 'Amon', 'pr')
    assert fix == [Pr(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.miroc5.Tas.fix_metadata',
    autospec=True)
def test_hur_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``hur``."""
    fix = Hur(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.miroc5.Tas.fix_metadata',
    autospec=True)
def test_pr_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``pr``."""
    fix = Pr(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


class TestSftof(unittest.TestCase):
    """Test sftof fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftof', units='J')
        self.fix = Sftof(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'MIROC5', 'Amon', 'sftof'),
                             [Sftof(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestTas(unittest.TestCase):
    """Test tas fixes."""

    def setUp(self):
        """Prepare tests."""
        self.coord_name = 'latitude'
        self.coord = iris.coords.DimCoord([3.141592],
                                          bounds=[[1.23, 4.5678910]],
                                          standard_name=self.coord_name)
        self.cube = Cube([1.0], dim_coords_and_dims=[(self.coord, 0)])
        self.fix = Tas(None)

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'MIROC5', 'Amon', 'tas'),
                             [Tas(None)])

    def test_fix_metadata(self):
        """Test metadata fix."""
        [cube] = self.fix.fix_metadata([self.cube])
        new_coord = self.coord.copy([3.14159], [[1.23, 4.56789]])
        self.assertEqual(cube.coord(self.coord_name), new_coord)
