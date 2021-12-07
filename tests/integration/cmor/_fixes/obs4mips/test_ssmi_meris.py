"""Test SSMI fixes."""
import unittest

from esmvalcore.cmor._fixes.obs4mips.ssmi_meris import Prw
from esmvalcore.cmor.fix import Fix


class TestPrw(unittest.TestCase):
    """Test prw fixes."""
    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('obs4MIPs', 'SSMI-MERIS', 'Amon', 'prw'),
            [Prw(None)])
