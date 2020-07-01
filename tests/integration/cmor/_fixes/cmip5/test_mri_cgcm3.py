"""Test MRI-CGCM3 fixes."""
import unittest

from esmvalcore.cmor._fixes.cmip5.mri_cgcm3 import Cl, Msftmyz, ThetaO
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


class TestMsftmyz(unittest.TestCase):
    """Test msftmyz fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'msftmyz'),
            [Msftmyz(None)])


class TestThetao(unittest.TestCase):
    """Test thetao fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'thetao'),
            [ThetaO(None)])
