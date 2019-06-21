import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.ec_earth import Sftlf, Sic


class TestSic(unittest.TestCase):
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sic', units='J')
        self.fix = Sic()

    def test_fix_data(self):
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))


class TestSftlf(unittest.TestCase):
    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='sftlf', units='J')
        self.fix = Sftlf()

    def test_fix_data(self):
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))
