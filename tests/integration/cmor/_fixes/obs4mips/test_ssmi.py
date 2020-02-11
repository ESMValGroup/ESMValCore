"""Test SSMI fixes."""
import unittest

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.obs4mips.ssmi import Prw


class TestPrw(unittest.TestCase):
    """Test prw fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(Fix.get_fixes('obs4mips', 'SSMI', 'Amon', 'prw'),
                             [Prw(None)])
