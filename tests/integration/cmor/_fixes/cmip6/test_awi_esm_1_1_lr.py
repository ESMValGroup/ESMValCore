"""Tests for the fixes of AWI-ESM-1-1-LR."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.awi_esm_1_1_lr import (
    AllVars,
    FesomSeaIceScalar,
)
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix

SEA_ICE_SCALAR_VARS = ["siextentn", "siextents", "siarean", "siareas"]


@pytest.fixture
def sample_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name="ta")
    tas_cube = iris.cube.Cube([3.0], var_name="tas")
    return iris.cube.CubeList([ta_cube, tas_cube])


@pytest.fixture(params=SEA_ICE_SCALAR_VARS)
def sea_ice_scalar_cubes(request):
    var_name = request.param
    time_coord = iris.coords.DimCoord(
        np.arange(3),  # random small number of timesteps
        standard_name="time",
    )
    cube_ok = iris.cube.Cube(
        np.ones((3,)),
        var_name=var_name,
        dim_coords_and_dims=[(time_coord, 0)],
    )
    nodes_coord = iris.coords.DimCoord(
        np.arange(1),
        var_name="nodes",
    )
    cube_bad = iris.cube.Cube(
        np.ones((1, 3)),
        var_name=var_name,
        dim_coords_and_dims=[(time_coord, 1), (nodes_coord, 0)],
    )
    return iris.cube.CubeList([cube_ok, cube_bad])


def test_get_tas_fix():
    fix = Fix.get_fixes("CMIP6", "AWI-ESM-1-1-LR", "Amon", "tas")
    assert fix == [AllVars(None), GenericFix(None)]


@pytest.mark.parametrize("short_name", SEA_ICE_SCALAR_VARS)
def test_get_sea_ice_scalar_fixes(short_name):
    fixes = Fix.get_fixes("CMIP6", "AWI-ESM-1-1-LR", "SImon", short_name)
    assert fixes == [FesomSeaIceScalar(None), AllVars(None), GenericFix(None)]


def test_allvars_fix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes["parent_time_units"] = "days since 0001-01-01 00:00:00"
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert (
            cube.attributes["parent_time_units"]
            == "days since 0001-01-01 00:00:00"
        )


def test_allvars_no_need_tofix_metadata(sample_cubes):
    for cube in sample_cubes:
        cube.attributes["parent_time_units"] = "days since 0001-01-01 00:00:00"
    out_cubes = AllVars(None).fix_metadata(sample_cubes)
    assert out_cubes is sample_cubes
    for cube in out_cubes:
        assert (
            cube.attributes["parent_time_units"]
            == "days since 0001-01-01 00:00:00"
        )


def test_sea_ice_scalar_fix_metadata(sea_ice_scalar_cubes):
    out_cubes = FesomSeaIceScalar(None).fix_metadata(sea_ice_scalar_cubes)
    for cube in out_cubes:
        assert cube.ndim == 1
