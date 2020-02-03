"""Tests for MIROC5."""
import unittest

import iris
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.miroc5 import Sftof, Tas
from esmvalcore.cmor.fix import Fix


class TestSftof(unittest.TestCase):
    """Test sftof fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftof', units='J')
        self.fix = Sftof()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'MIROC5', 'sftof'),
                             [Sftof()])

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
        self.fix = Tas()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'MIROC5', 'tas'), [Tas()])

    def test_fix_metadata(self):
        """Test metadata fix."""
        [cube] = self.fix.fix_metadata([self.cube])
        new_coord = self.coord.copy([3.14159], [[1.23, 4.56789]])
        self.assertEqual(cube.coord(self.coord_name), new_coord)
