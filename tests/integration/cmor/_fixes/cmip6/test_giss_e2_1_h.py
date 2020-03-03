"""Test fixes for GISS-E2-1-H."""
import unittest

from esmvalcore.cmor._fixes.cmip6.giss_e2_1_h import Cl
from esmvalcore.cmor._fixes.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GISS-E2-1-H', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.giss_e2_1_h.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``cl``."""
    fix = Cl(None)
    fix.fix_metadata('cubes')
    mock_base_fix_metadata.assert_called_once_with(fix, 'cubes')
