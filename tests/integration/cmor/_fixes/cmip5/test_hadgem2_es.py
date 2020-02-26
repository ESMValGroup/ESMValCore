"""Test HADGEM2-ES fixes."""
import unittest

from esmvalcore.cmor._fixes.cmip5.hadgem2_es import O2, AllVars
from esmvalcore.cmor.fix import Fix


class TestAllVars(unittest.TestCase):
    """Test allvars fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'HADGEM2-ES', 'Amon', 'tas'),
            [AllVars(None)])


class TestO2(unittest.TestCase):
    """Test o2 fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'HADGEM2-ES', 'Amon', 'o2'),
            [O2(None), AllVars(None)])
