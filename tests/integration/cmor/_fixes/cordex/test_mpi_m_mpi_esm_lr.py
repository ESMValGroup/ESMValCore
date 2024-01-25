"""Tests for the fixes of driver MPI-M-MPI-ESM-LR."""
import pytest

from esmvalcore.cmor.fix import Fix


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_regcm4_6_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'REGCM4-6',
        'Amon',
        short_name,
        extra_facets={'driver': 'MPI-M-MPI-ESM-LR'})
    assert isinstance(fix[0], Fix)


def test_get_racmo22e_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'RACMO22E',
        'Amon',
        'pr',
        extra_facets={'driver': 'MPI-M-MPI-ESM-LR'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'HadREM3-GA7-05',
        'Amon',
        short_name,
        extra_facets={'driver': 'MPI-M-MPI-ESM-LR'})
    assert isinstance(fix[0], Fix)
