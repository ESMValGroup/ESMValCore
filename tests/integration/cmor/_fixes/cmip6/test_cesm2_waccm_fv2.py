"""Tests for the fixes of CESM2-WACCM-FV2."""

from esmvalcore.cmor._fixes.cmip6.cesm2 import Fgco2 as BaseFgco2
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm_fv2 import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Siconc,
    Pr,
    Tas,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
import iris
import numpy as np
import pytest


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is BaseCl


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "Amon", "cli")
    assert fix == [Cli(None), GenericFix(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is BaseCl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "Amon", "clw")
    assert fix == [Clw(None), GenericFix(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is BaseCl


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "Omon", "fgco2")
    assert fix == [Fgco2(None), Omon(None), GenericFix(None)]


def test_fgco2_fix():
    """Test fix for ``fgco2``."""
    assert Fgco2 is BaseFgco2


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is SiconcFixScalarCoord


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM-FV2", "Amon", "tas")
    assert fix == [Tas(None), GenericFix(None)]


def test_tas_fix():
    """Test fix for ``tas``."""
    assert Tas is BaseTas


@pytest.fixture
def pr_cubes():
    wrong_time_coord = iris.coords.AuxCoord(
        points=[1.0, 2.0, 1.0, 2.0, 3.0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01",
    )

    correct_time_coord = iris.coords.AuxCoord(
        points=[1.0, 2.0, 3.0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01",
    )

    correct_lat_coord = iris.coords.DimCoord(
        [0.0], var_name="lat", standard_name="latitude"
    )
    wrong_lat_coord = iris.coords.DimCoord(
        [0.0], var_name="latitudeCoord", standard_name="latitude"
    )
    correct_lon_coord = iris.coords.DimCoord(
        [0.0], var_name="lon", standard_name="longitude"
    )
    wrong_lon_coord = iris.coords.DimCoord(
        [0.0], var_name="longitudeCoord", standard_name="longitude"
    )

    wrong_coord_specs = [
        (wrong_time_coord, 0),
        (wrong_lat_coord, 1),
        (wrong_lon_coord, 2),
    ]

    correct_coord_specs = [
        (correct_time_coord, 0),
        (correct_lat_coord, 1),
        (correct_lon_coord, 2),
    ]
    correct_pr_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name="pr",
        dim_coords_and_dims=correct_coord_specs,
    )

    wrong_pr_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name="ta",
        dim_coords_and_dims=wrong_coord_specs,
    )

    return iris.cube.CubeList([correct_pr_cube, wrong_pr_cube])


def test_get(self):
    """Test fix get."""
    self.assertListEqual(
        Fix.get_fixes("CMIP6", "CESM2", "day", "pr"),
        [Pr(None), GenericFix(None)],
    )


def test_pr_fix_metadata(self):
    """Test metadata fix."""
    out_wrong_cube = self.fix.fix_metadata(self.wrong_cube)
    out_correct_cube = self.fix.fix_metadata(self.correct_cube)

    time = out_wrong_cube[0].coord("time")
    assert time == self.time_coord

    time = out_correct_cube[0].coord("time")
    assert time == self.time_coord


def test_pr_fix_metadata_no_time(self):
    """Test metadata fix with no time coord."""
    self.correct_cube[0].remove_coord("time")
    out_correct_cube = self.fix.fix_metadata(self.correct_cube)
    with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
        out_correct_cube[0].coord("time")