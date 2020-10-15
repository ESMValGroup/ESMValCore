"""Test MRI-ESM1 fixes."""
import unittest

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.mri_esm1 import Msftmyz


class TestMsftmyz(unittest.TestCase):
    """Test msftmyz fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-ESM1', 'Amon', 'msftmyz'),
            [Msftmyz(None)])
