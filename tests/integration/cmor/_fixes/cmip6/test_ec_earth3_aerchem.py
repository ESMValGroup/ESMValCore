"""Tests for EC-Earth3-AerChem model."""

import iris
import pytest

from esmvalcore.cmor._fixes.cmip6.ec_earth3_aerchem import Oh
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def oh_cubes():
    air_pressure_coord = iris.coords.DimCoord(
        [1000.09, 600.6, 200.0],
        bounds=[[1200.00001, 800], [800, 400.8], [400.8, 1.9]],
        var_name="ps",
        standard_name="air_pressure",
        units="pa",
    )
    oh_cube = iris.cube.Cube(
        [0.0, 1.0, 2.0],
        var_name="oh",
        dim_coords_and_dims=[(air_pressure_coord, 0)],
    )
    return iris.cube.CubeList([oh_cube])


def test_get_oh_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "EC-Earth3-AerChem", "AERmonZ", "oh")
    assert Oh(None) in fix


def test_oh_fix_metadata(oh_cubes):
    """Test ``fix_metadata`` for ``oh``."""
    vardef = get_var_info("CMIP6", "AERmonZ", "oh")
    fix = Oh(vardef)
    fixed_cubes = fix.fix_metadata(oh_cubes)
    for coord in fixed_cubes[0].coords():
        if coord.var_name == "ps":
            assert coord.standard_name == "surface_air_pressure"
