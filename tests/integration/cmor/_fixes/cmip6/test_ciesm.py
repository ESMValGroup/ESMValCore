"""Tests for the fixes of CIESM."""
import iris.cube
import pytest

from esmvalcore.cmor._fixes.cmip6.ciesm import Cl
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'CIESM', 'Amon', 'cl')
    assert fix == [Cl(None)]


@pytest.fixture
def cl_cube():
    """``cl`` cube."""
    cube = iris.cube.Cube(
        [1.0],
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        units='%',
    )
    return cube


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, ClFixHybridPressureCoord)


def test_cl_fix_data(cl_cube):
    """Test ``fix_data`` for ``cl``."""
    fix = Cl(None)
    out_cube = fix.fix_data(cl_cube)
    assert out_cube.data == [100.0]


def test_pr_fix():
    """Test `Pr.fix_metadata`."""
    cube = iris.cube.Cube(
        [2.82e-08],
        var_name='pr',
        units='kg m-2 s-1',
    )

    fix = Fix.get_fixes('CMIP6', 'CIESM', 'Amon', 'pr')[0]

    out_cube = fix.fix_data(cube)
    assert out_cube.data == [2.82e-05]
    assert out_cube.units == 'kg m-2 s-1'
