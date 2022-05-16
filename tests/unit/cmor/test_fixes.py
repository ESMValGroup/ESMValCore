"""Test individual fix functions."""
import pytest

import esmvalcore.cmor._fixes.shared as shared
import esmvalcore.cmor.fixes as fixes


@pytest.mark.parametrize('func', [
    'add_altitude_from_plev',
    'add_plev_from_altitude',
])
def test_imports(func):
    assert func in fixes.__all__
    fn_in_shared = getattr(shared, func)
    fn_in_fixes = getattr(fixes, func)
    assert fn_in_shared is fn_in_fixes
