"""Tests for the fixes of AWI-CM-1-1-MR."""
import pytest

from esmvalcore.cmor.fix import Fix


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_clmcom_cclm4_8_17fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'CLMCOM-CCLM4-8-17',
        'Amon',
        short_name,
        extra_facets={'driver': 'MIROC-MIROC5'})
    assert isinstance(fix[0], Fix)


def test_get_gerics_remo2015_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'GERICS-REMO2015',
        'Amon',
        'pr',
        extra_facets={'driver': 'MIROC-MIROC5'})
    assert isinstance(fix[0], Fix)
