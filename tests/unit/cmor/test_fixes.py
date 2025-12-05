"""Test individual fix functions."""

import pytest

from esmvalcore.cmor import fixes
from esmvalcore.cmor._fixes import shared


@pytest.mark.parametrize(
    "func",
    [
        "add_altitude_from_plev",
        "add_plev_from_altitude",
        "get_next_month",
        "get_time_bounds",
    ],
)
def test_imports(func):
    assert func in fixes.__all__
    fn_in_shared = getattr(shared, func)
    fn_in_fixes = getattr(fixes, func)
    assert fn_in_shared is fn_in_fixes
