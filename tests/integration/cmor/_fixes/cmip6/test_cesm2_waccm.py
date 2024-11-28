"""Tests for the fixes of CESM2-WACCM."""

import os
import unittest.mock

import iris
import numpy as np
import pytest
import pandas as pd

from esmvalcore.cmor._fixes.cmip6.cesm2 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cesm2 import Fgco2 as BaseFgco2
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Siconc,
    Tas,
    Pr,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, BaseCl)


@unittest.mock.patch(
    "esmvalcore.cmor._fixes.cmip6.cesm2.Fix.get_fixed_filepath", autospec=True
)
def test_cl_fix_file(mock_get_filepath, tmp_path, test_data_path):
    """Test ``fix_file`` for ``cl``."""
    nc_path = test_data_path / "cesm2_waccm_cl.nc"
    mock_get_filepath.return_value = os.path.join(
        tmp_path, "fixed_cesm2_waccm_cl.nc"
    )
    fix = Cl(None)
    fixed_file = fix.fix_file(nc_path, tmp_path)
    mock_get_filepath.assert_called_once_with(
        tmp_path, nc_path, add_unique_suffix=False
    )
    fixed_cube = iris.load_cube(fixed_file)
    lev_coord = fixed_cube.coord(var_name="lev")
    a_coord = fixed_cube.coord(var_name="a")
    b_coord = fixed_cube.coord(var_name="b")
    assert lev_coord.standard_name == (
        "atmosphere_hybrid_sigma_pressure_coordinate"
    )
    assert lev_coord.units == "1"
    np.testing.assert_allclose(a_coord.points, [1.0, 2.0])
    np.testing.assert_allclose(a_coord.bounds, [[0.0, 1.5], [1.5, 3.0]])
    np.testing.assert_allclose(b_coord.points, [0.0, 1.0])
    np.testing.assert_allclose(b_coord.bounds, [[-1.0, 0.5], [0.5, 2.0]])


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Omon", "fgco2")
    assert fix == [Fgco2(None), Omon(None), GenericFix(None)]


def test_fgco2_fix():
    """Test fix for ``fgco2``."""
    assert Fgco2 is BaseFgco2


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord


@pytest.fixture
def tas_cubes():
    """Cubes to test fixes for ``tas``."""
    ta_cube = iris.cube.Cube([1.0], var_name="ta")
    tas_cube = iris.cube.Cube([3.0], var_name="tas")
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Amon", "tas")
    assert fix == [Tas(None), GenericFix(None)]


def test_tas_fix():
    """Test fix for ``tas``."""
    assert Tas is BaseTas


@pytest.fixture
def pr_cubes():
    correct_time_coord = iris.coords.DimCoord(
        points=[1.0, 2.0, 3.0, 4.0, 5.0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01",
    )

    lat_coord = iris.coords.DimCoord(
        [0.0], var_name="lat", standard_name="latitude"
    )

    lon_coord = iris.coords.DimCoord(
        points=[0.0], var_name="lon", standard_name="longitude"
    )

    correct_coord_specs = [
        (correct_time_coord, 0),
        (lat_coord, 1),
        (lon_coord, 2),
    ]

    correct_pr_cube = iris.cube.Cube(
        np.ones((5, 1, 1)),
        var_name="pr",
        units="kg m-2 s-1",
        dim_coords_and_dims=correct_coord_specs,
    )

    scalar_cube = iris.cube.Cube(0.0, var_name="ps")

    return iris.cube.CubeList([correct_pr_cube, scalar_cube])


def test_get_pr_fix():
    """Test pr fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2", "day", "pr")
    assert fix == [Pr(None), GenericFix(None)]


def test_pr_fix_metadata(pr_cubes):
    """Test metadata fix."""
    vardef = get_var_info("CMIP6", "day", "pr")
    fix = Pr(vardef)

    out_cubes = fix.fix_metadata(pr_cubes)
    assert out_cubes[0].var_name == "pr"
    coord = out_cubes[0].coord("time")
    assert pd.Series(coord.points).is_monotonic_increasing