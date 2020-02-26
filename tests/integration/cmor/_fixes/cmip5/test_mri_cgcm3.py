"""Test MRI-GCM3 fixes."""
import unittest

from esmvalcore.cmor._fixes.cmip5.mri_cgcm3 import Msftmyz, ThetaO
from esmvalcore.cmor.fix import Fix


class TestMsftmyz(unittest.TestCase):
    """Test msftmyz fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'msftmyz'),
            [Msftmyz(None)])


class TestThetao(unittest.TestCase):
    """Test thetao fixes."""
    def test_get(self):
        """Test fix get"""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'thetao'),
            [ThetaO(None)])
