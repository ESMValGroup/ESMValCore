"""Tests for the fixes for driver ICHEC-EC-Earth."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cordex.ichec_ec_earth import (
    cosmo_crclim_v1_1,
    hadrem3_ga7_05,
    wrf381p,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_remo2015_fix():
    fix = Fix.get_fixes(
        "CORDEX",
        "REMO2015",
        "Amon",
        "pr",
        extra_facets={"driver": "ICHEC-EC-Earth"},
    )
    assert isinstance(fix[0], Fix)


def test_get_racmo22e_fix():
    fix = Fix.get_fixes(
        "CORDEX",
        "RACMO22E",
        "Amon",
        "pr",
        extra_facets={"driver": "ICHEC-EC-Earth"},
    )
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "HadREM3-GA7-05",
        "Amon",
        short_name,
        extra_facets={"driver": "ICHEC-EC-Earth"},
    )
    assert isinstance(fix[0], Fix)


def test_hadrem3_ga7_05_sic() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "HadREM3-GA7-05",
        "day",
        "sic",
        extra_facets={"driver": "CNRM-CERFACS-CNRM-CM5"},
    )
    assert any(isinstance(fix, hadrem3_ga7_05.Sic) for fix in fixes)

    cube = iris.cube.Cube(
        np.array([0.5], dtype=np.float32),
        var_name="sic",
        standard_name="sea_ice_area_fraction",
        units="%",
    )
    fix = next(fix for fix in fixes if isinstance(fix, hadrem3_ga7_05.Sic))
    result = fix.fix_metadata([cube])
    assert result[0].units == "%"
    np.testing.assert_allclose(result[0].data, [50.0])


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_rca4_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "RCA4",
        "Amon",
        short_name,
        extra_facets={"driver": "ICHEC-EC-Earth"},
    )
    assert isinstance(fix[0], Fix)


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
        extra_facets={"driver": "ICHEC-EC-Earth"},
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


def test_get_cosmo_crclim_v1_1_fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "COSMO-crCLIM-v1-1",
        "day",
        "snw",
        extra_facets={"driver": "ICHEC-EC-Earth"},
    )
    assert any(isinstance(fix, cosmo_crclim_v1_1.Snw) for fix in fixes)
