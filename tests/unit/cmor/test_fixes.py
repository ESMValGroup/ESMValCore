"""Test individual fix functions."""
import pytest

import esmvalcore.cmor.fixes


@pytest.mark.parametrize('func', [
    'add_plev_from_altitude',
    'add_sigma_factory',
])
def test_imports(func):
    assert func in esmvalcore.cmor.fixes.__all__
