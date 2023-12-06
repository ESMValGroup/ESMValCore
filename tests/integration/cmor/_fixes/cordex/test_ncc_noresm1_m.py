"""Tests for the fixes of driver NCC-NorESM1-M."""
import pytest

from esmvalcore.cmor.fix import Fix


def test_get_remo2015_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'REMO2015',
        'Amon',
        'pr',
        extra_facets={'driver': 'NCC-NorESM1-M'})
    assert isinstance(fix[0], Fix)


def test_get_racmo22e_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'RACMO22E',
        'Amon',
        'pr',
        extra_facets={'driver': 'NCC-NorESM1-M'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'NCC-NorESM1-M'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_rca4_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'RCA4',
        'Amon',
        short_name,
        extra_facets={'driver': 'NCC-NorESM1-M'})
    assert isinstance(fix[0], Fix)
