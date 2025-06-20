"""Test fixes for EC-Earth3-Veg-LR."""

import iris.coords
import iris.cube
import pytest

from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg_lr import AllVars, Siconc
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import Fix


def test_get_siconc_fix():
    """Test getting of fix."""
    assert Siconc(None) in Fix.get_fixes(
        "CMIP6",
        "EC-Earth3-Veg-LR",
        "SImon",
        "siconc",
    )


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is OceanFixGrid


def test_get_allvars_fix():
    """Test getting of fix."""
    assert AllVars(None) in Fix.get_fixes(
        "CMIP6",
        "EC-Earth3-Veg-LR",
        "Omon",
        "tos",
    )


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
