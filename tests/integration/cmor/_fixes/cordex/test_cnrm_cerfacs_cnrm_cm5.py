"""Tests for the fixes for driver CNRM-CERFACS-CNRM-CM5."""

import iris
import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5 import (
    aladin53,
    aladin63,
    wrf381p,
)
from esmvalcore.cmor._fixes.cordex.cordex_fixes import CLMcomCCLM4817
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
    correct_height_coord = iris.coords.AuxCoord([2.0], var_name="height")
    wrong_height_coord = iris.coords.AuxCoord([10.0], var_name="height")
    correct_cube = iris.cube.Cube(
        [10.0],
        var_name="tas",
        dim_coords_and_dims=[(correct_time_coord, 0)],
        aux_coords_and_dims=[(correct_height_coord, ())],
    )
    wrong_cube = iris.cube.Cube(
        [10.0],
        var_name="tas",
        dim_coords_and_dims=[(correct_time_coord, 0)],
        aux_coords_and_dims=[(wrong_height_coord, ())],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "HadREM3-GA7-05",
        "mon",
        short_name,
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert isinstance(fix[0], Fix)


def test_fix_aladin53_sftlf() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "ALADIN53",
        "fx",
        "sftlf",
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert isinstance(fixes[0], aladin53.Sftlf)
    cube = iris.cube.Cube(
        [0, 1.0],
        var_name="sftlf",
        units="%",
    )
    (result,) = fixes[0].fix_metadata([cube])
    assert result.data.tolist() == [0, 100.0]
    assert result.units == "%"


def test_fix_aladin53_ts() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "ALADIN53",
        "day",
        "ts",
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert isinstance(fixes[0], aladin53.Ts)
    cube = iris.cube.Cube(
        [0, 1.0],
        var_name="ts",
        units="K",
    )
    (result,) = fixes[0].fix_metadata([cube])
    assert result.data.tolist() == [273.15, 274.15]
    assert result.units == "K"


def test_fix_aladin53_tas() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "ALADIN53",
        "day",
        "tas",
        extra_facets={
            "driver": "CNRM-CERFACS-CNRM-CM5",
            "domain": "EUR-11",
        },
    )
    assert isinstance(fixes[0], aladin53.AllVars)
    wrong_coord_system = iris.coord_systems.LambertConformal(
        central_lat=49.5,
        central_lon=10.5,
        secant_latitudes=(49.5,),
        false_easting=400000.0,
        false_northing=-100000.0,
    )
    cube = iris.cube.Cube(
        np.array([0, 1.0]).reshape(1, 2),
        var_name="tas",
        units="K",
        aux_coords_and_dims=[
            (
                iris.coords.AuxCoord(
                    [0.0],
                    standard_name="projection_y_coordinate",
                    units="m",
                    coord_system=wrong_coord_system,
                ),
                (0,),
            ),
            (
                iris.coords.AuxCoord(
                    [0.0, 1.0],
                    standard_name="projection_x_coordinate",
                    units="m",
                    coord_system=wrong_coord_system,
                ),
                (1,),
            ),
        ],
    )
    (result,) = fixes[0].fix_metadata([cube])
    assert result.coord_system() == iris.coord_systems.LambertConformal(
        central_lat=49.5,
        central_lon=10.5,
        secant_latitudes=(49.5,),
        false_easting=0,
        false_northing=0,
    )


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_aladin63_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "ALADIN63",
        "mon",
        short_name,
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert isinstance(fix[0], Fix)


def test_aladin63_height_fix(cubes):
    fix = aladin63.Tas(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord("height").points == 2.0


@pytest.mark.parametrize(
    "short_name",
    ["tasmax", "tasmin", "tas", "hurs", "huss"],
)
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "WRF381P",
        "mon",
        short_name,
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
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
    vardef = get_var_info("CMIP6", "Amon", "tas")
    fix = wrf381p.Tas(vardef)
    out_cubes = fix.fix_metadata([cube])
    assert out_cubes[0].coord("height").points == 2.0


def test_get_cclm4_8_17fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "CCLM4-8-17",
        "mon",
        "ts",
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert any(isinstance(fix, CLMcomCCLM4817) for fix in fixes)
