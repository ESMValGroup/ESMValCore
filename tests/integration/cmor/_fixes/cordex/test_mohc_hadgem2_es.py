"""Tests for the fixes for driver MOHC-HadGEM2-ES."""

import iris
import iris.coords
import iris.cube
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cordex.cordex_fixes import CLMcomCCLM4817
from esmvalcore.cmor._fixes.cordex.mohc_hadgem2_es import (
    cclm4_8_17,
    cosmo_crclim_v1_1,
    hirham5,
    wrf381p,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord(
        [0.0],
        var_name="time",
        standard_name="time",
        long_name="time",
    )
    wrong_time_coord = iris.coords.DimCoord(
        [0.0],
        var_name="time",
        standard_name="time",
        long_name="wrong",
    )
    correct_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
        long_name="latitude",
    )
    wrong_lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="latitudeCoord",
        standard_name="latitude",
        long_name="latitude",
        attributes={"wrong": "attr"},
    )
    correct_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
        long_name="longitude",
    )
    wrong_lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="longitudeCoord",
        standard_name="longitude",
        long_name="longitude",
        attributes={"wrong": "attr"},
    )
    correct_cube = iris.cube.Cube(
        [[[10.0]]],
        var_name="tas",
        dim_coords_and_dims=[
            (correct_time_coord, 0),
            (correct_lat_coord, 1),
            (correct_lon_coord, 2),
        ],
    )
    wrong_cube = iris.cube.Cube(
        [[[10.0]]],
        var_name="tas",
        dim_coords_and_dims=[
            (wrong_time_coord, 0),
            (wrong_lat_coord, 1),
            (wrong_lon_coord, 2),
        ],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_hirham5_fix():
    fix = Fix.get_fixes(
        "CORDEX",
        "HIRHAM5",
        "Amon",
        "pr",
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_remo2015_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "REMO2015",
        "Amon",
        short_name,
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "HadREM3-GA7-05",
        "Amon",
        short_name,
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_rca4_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "RCA4",
        "Amon",
        short_name,
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert isinstance(fix[0], Fix)


def test_hirham5_fix(cubes):
    fix = hirham5.Pr(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord("latitude").attributes == {}
        assert cube.coord("longitude").attributes == {}


@pytest.mark.parametrize(
    "short_name",
    ["tasmax", "tasmin", "tas", "hurs", "huss"],
)
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "WRF381P",
        "Amon",
        short_name,
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert isinstance(fix[0], Fix)


def test_wrf381p_height_fix():
    time_coord = iris.coords.DimCoord(
        [0.0],
        var_name="time",
        standard_name="time",
        long_name="time",
    )
    cube = iris.cube.Cube(
        [10.0],
        var_name="tas",
        dim_coords_and_dims=[(time_coord, 0)],
    )
    vardef = get_var_info("CORDEX", "day", "tas")
    fix = wrf381p.Tas(vardef)
    out_cubes = fix.fix_metadata([cube])
    assert out_cubes[0].coord("height").points == 2.0


def test_get_cclm4_8_17fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "CCLM4-8-17",
        "Amon",
        "ts",
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert any(isinstance(fix, CLMcomCCLM4817) for fix in fixes)


def test_cclm4_8_17_clivi() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "CCLM4-8-17",
        "day",
        "clivi",
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert any(isinstance(fix, cclm4_8_17.Clivi) for fix in fixes)

    cube = iris.cube.Cube(
        np.array([1.8e-5], dtype=np.float32),
        var_name="clivi",
        standard_name="atmosphere_cloud_ice_content",
        units="kg m-2",
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    [0.0],
                    var_name="time",
                ),
                0,
            ),
        ],
    )
    fix = next(fix for fix in fixes if isinstance(fix, cclm4_8_17.Clivi))
    result = fix.fix_metadata([cube])
    assert result[0].units == "kg m-2"
    np.testing.assert_allclose(result[0].data, [1.8e-2])


def test_cclm4_8_17_prw() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "CCLM4-8-17",
        "day",
        "prw",
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert any(isinstance(fix, cclm4_8_17.Prw) for fix in fixes)

    cube = iris.cube.Cube(
        np.array([0.014], dtype=np.float32),
        var_name="prw",
        standard_name="atmosphere_water_vapor_content",
        units="kg m-2",
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    [0.0],
                    var_name="time",
                ),
                0,
            ),
        ],
    )
    fix = next(fix for fix in fixes if isinstance(fix, cclm4_8_17.Prw))
    result = fix.fix_metadata([cube])
    assert result[0].units == "kg m-2"
    np.testing.assert_allclose(result[0].data, [14.0])


def test_get_cosmo_crclim_v1_1_fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "COSMO-crCLIM-v1-1",
        "day",
        "snw",
        extra_facets={"driver": "MOHC-HadGEM2-ES"},
    )
    assert any(isinstance(fix, cosmo_crclim_v1_1.Snw) for fix in fixes)
