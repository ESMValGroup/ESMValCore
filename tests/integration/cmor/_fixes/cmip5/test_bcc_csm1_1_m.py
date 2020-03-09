"""Test Access1-0 fixes."""
import unittest

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1_m import Tos


class TestTos(unittest.TestCase):
    """Test tos fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'bcc-csm1-1-m', 'Amon', 'tos'), [Tos(None)])


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip5.bcc_csm1_1_m.BaseTos.fix_data',
    autospec=True)
def test_tos_fix_data(mock_base_fix_data):
    """Test ``fix_data`` for ``tos``."""
    fix = Tos(None)
    fix.fix_data('cubes')
    mock_base_fix_data.assert_called_once_with(fix, 'cubes')
