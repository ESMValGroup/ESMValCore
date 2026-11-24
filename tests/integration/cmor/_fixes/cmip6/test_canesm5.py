"""Tests for the fixes of CanESM5."""

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.canesm5 import Cl, Cli, Clw, Co2, Gpp
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_co2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5", "Amon", "co2")
    assert fix == [Co2(None), GenericFix(None)]


@pytest.fixture
def co2_cube():
    """``co2`` cube."""
    return iris.cube.Cube(
        [1.0],
        var_name="co2",
        standard_name="mole_fraction_of_carbon_dioxide_in_air",
        units="mol mol-1",
    )


def test_co2_fix_data(co2_cube):
    """Test ``fix_data`` for ``co2``."""
    fix = Co2(None)
    out_cube = fix.fix_data(co2_cube)
    np.testing.assert_allclose(out_cube.data, [1.0e-6])


@pytest.fixture
def gpp_cube():
    """``gpp`` cube."""
    return iris.cube.Cube(
        [0, 1],
        var_name="gpp",
        standard_name="gross_primary_productivity_of_biomass_expressed_as_"
        "carbon",
        units="kg m-2 s-1",
    )


def test_get_gpp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5", "Lmon", "gpp")
    assert fix == [Gpp(None), GenericFix(None)]


def test_gpp_fix_data(gpp_cube):
    """Test ``fix_data`` for ``gpp``."""
    fix = Gpp(None)
    out_cube = fix.fix_data(gpp_cube)
    np.testing.assert_allclose(
        out_cube.data,
        np.ma.masked_invalid([np.nan, 1]),
    )
    assert np.array_equal(out_cube.data.mask, [True, False])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CanESM5", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_get_ps_fix():
    """Test that ``ps`` files, which have no hybrid coordinate, are not fixed."""
    assert Fix.get_fixes("CMIP6", "CanESM5", "Amon", "ps") == [
        GenericFix(None),
    ]


@pytest.mark.parametrize("fix_class", [Cl, Cli, Clw])
def test_cl_fix_metadata(fix_class):
    """Test that the standard name of ``ps`` no longer clashes."""
    ps_coord = iris.coords.AuxCoord(
        [0.0],
        var_name="ps",
        standard_name="air_pressure",
    )
    cube = iris.cube.Cube(
        [1.0],
        var_name="cl",
        standard_name="cloud_area_fraction_in_atmosphere_layer",
        aux_coords_and_dims=[(ps_coord, 0)],
    )
    vardef = get_var_info("CMIP6", "Amon", "cl")
    fixed_cubes = fix_class(vardef).fix_metadata(iris.cube.CubeList([cube]))
    assert fixed_cubes[0].coord(var_name="ps").standard_name is None
