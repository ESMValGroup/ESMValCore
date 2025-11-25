"""Tests for the fixes of NorESM2-MM."""

import iris.coords
import iris.cube
import pytest

from esmvalcore.cmor._fixes.cmip6.noresm2_mm import AllVars, Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix


def test_get_allvars_fix():
    """Test getting of fix."""
    assert AllVars(None) in Fix.get_fixes("CMIP6", "NorESM2-MM", "Omon", "tos")


@pytest.mark.parametrize("has_index_coord", [True, False])
def test_grid_fix(has_index_coord):
    """Test fix for differing grid index coordinate long names."""
    cube = iris.cube.Cube([1, 2])
    if has_index_coord:
        i_coord = iris.coords.DimCoord(
            [0.0, 1.0],
            var_name="i",
            long_name="first spatial index for variables stored on an unstructured grid",
            units=1,
        )
        cube.add_dim_coord(i_coord, 0)

    cubes = [cube]
    for fix in Fix.get_fixes("CMIP6", "NorESM2-MM", "Omon", "tos"):
        cubes = fix.fix_metadata(cubes)

    assert len(cubes) == 1
    if has_index_coord:
        assert (
            cubes[0].coord(var_name="i").long_name
            == "cell index along first dimension"
        )


def test_get_cl_fix():
    """Test getting of fix."""
    assert Cl(None) in Fix.get_fixes("CMIP6", "NorESM2-MM", "Amon", "cl")


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    assert Cli(None) in Fix.get_fixes("CMIP6", "NorESM2-MM", "Amon", "cli")


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    assert Clw(None) in Fix.get_fixes("CMIP6", "NorESM2-MM", "Amon", "clw")


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord
