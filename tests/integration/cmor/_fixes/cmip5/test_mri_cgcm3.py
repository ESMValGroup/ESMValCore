"""Test MRI-GCM3 fixes."""
import unittest

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor._fixes.cmip5.mri_cgcm3 import Msftmyz, ThetaO


class TestMsftmyz(unittest.TestCase):
    """Test msftmyz fixes."""

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'msftmyz'), [Msftmyz()]
        )


class TestThetao(unittest.TestCase):
    """Test thetao fixes."""

    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'thetao'), [ThetaO()]
        )
