"""Tests for the fixes of HadGEM3-GC31-LL."""
import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.hadgem3_gc31_ll import AllVars, Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridHeightCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def sample_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'HadGEM3-GC31-LL', 'Amon', 'tas')
    assert fix == [AllVars(None), GenericFix(None)]


def test_allvars_fix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'


def test_allvars_no_need_tofix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes['parent_time_units'] = 'days since 1850-01-01'
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert cube.attributes['parent_time_units'] == 'days since 1850-01-01'


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'HadGEM3-GC31-LL', 'Amon', 'cl')
    assert fix == [Cl(None), AllVars(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridHeightCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'HadGEM3-GC31-LL', 'Amon', 'cli')
    assert fix == [Cli(None), AllVars(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridHeightCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'HadGEM3-GC31-LL', 'Amon', 'clw')
    assert fix == [Clw(None), AllVars(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridHeightCoord
