"""Tests for the fixes of driver NCC-NorESM1-M."""
import pytest

from esmvalcore.cmor.fix import Fix


@pytest.mark.parametrize('short_name', ['tasmax', 'tasmin', 'tas'])
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'WRF381P',
        'Amon',
        short_name,
        extra_facets={'driver': 'IPSL-CM5A-MR'})
    assert isinstance(fix[0], Fix)
