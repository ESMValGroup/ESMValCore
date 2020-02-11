"""Test Access1-0 fixes."""
import unittest

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1_m import Tos


class TestTos(unittest.TestCase):
    """Test tos fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'BCC-CSM1-1-M', 'Amon', 'tos'), [Tos(None)])
