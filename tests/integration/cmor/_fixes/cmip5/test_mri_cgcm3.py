"""Test MRI-CGCM3 fixes."""
import unittest

from esmvalcore.cmor._fixes.cmip5.mri_cgcm3 import Cl, Msftmyz, ThetaO
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'MRI-CGCM3', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.mri_cgcm3.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cl``."""
    fix = Cl(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


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
