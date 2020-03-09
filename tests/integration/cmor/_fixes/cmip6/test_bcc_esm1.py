"""Test fixes for BCC-ESM1."""
import unittest

from esmvalcore.cmor._fixes.cmip6.bcc_esm1 import Cl, Tos
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.bcc_esm1.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cl``."""
    fix = Cl(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-ESM1', 'Omon', 'tos')
    assert fix == [Tos(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.bcc_esm1.BaseTos.fix_metadata',
    autospec=True)
def test_tos_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``tos``."""
    fix = Tos(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')
