"""Tests for the fixes for driver ICHEC-EC-Earth."""
import pytest

from esmvalcore.cmor.fix import Fix


def test_get_gerics_remo2015_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'GERICS-REMO2015',
        'Amon',
        'pr',
        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


def test_get_knmi_racmo22e_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'KNMI-RACMO22E',
        'Amon',
        'pr',
        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_mohc_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'MOHC-HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_smhi_rca4_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'SMHI-RCA4',
        'Amon',
        short_name,
        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)
