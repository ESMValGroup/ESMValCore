"""Test MPI-ESM-LR fixes."""
import unittest

from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.mpi_esm_lr import Pctisccp


class TestPctisccp2(unittest.TestCase):
    """Test Pctisccp2 fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='pctisccp', units='J')
        self.fix = Pctisccp()

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MPI-ESM-LR', 'pctisccp'), [Pctisccp()])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('J'))
