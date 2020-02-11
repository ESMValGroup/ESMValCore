"""Tests for inmcm4 fixes."""
import unittest

from iris.cube import Cube
from cf_units import Unit

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.inmcm4 import Gpp, Lai, Nbp


class TestGpp(unittest.TestCase):
    """Test gpp fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='gpp', units='J')
        self.fix = Gpp(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'INMCM4', 'Amon', 'gpp'),
                             [Gpp(None)])

    def test_fix_data(self):
        """Test data fox."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], -1)
        self.assertEqual(cube.units, Unit('J'))


class TestLai(unittest.TestCase):
    """Test lai fixes."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='lai', units='J')
        self.fix = Lai(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'INMCM4', 'Amon', 'lai'),
                             [Lai(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 1.0 / 100.0)
        self.assertEqual(cube.units, Unit('J'))


class TestNbp(unittest.TestCase):
    """Tests for nbp."""
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='nbp')
        self.fix = Nbp(None)

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('CMIP5', 'INMCM4', 'Amon', 'nbp'),
                             [Nbp(None)])

    def test_fix_metadata(self):
        """Test fix on nbp files to set standard_name."""
        new_cube = self.fix.fix_metadata([self.cube])[0]
        self.assertEqual(
            new_cube.standard_name,
            'surface_net_downward_mass_flux_of_carbon_dioxide_'
            'expressed_as_carbon_due_to_all_land_processes')
