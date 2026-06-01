"""Tests for the fixes of CESM2-WACCM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.cesm2 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.cesm2 import Fgco2 as BaseFgco2
from esmvalcore.cmor._fixes.cmip6.cesm2 import Pr as BasePr
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tas as BaseTas
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tasmax as BaseTasmax
from esmvalcore.cmor._fixes.cmip6.cesm2 import Tasmin as BaseTasmin
from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import (
    Cl,
    Cli,
    Clw,
    Fgco2,
    Omon,
    Pr,
    Siconc,
    Tas,
    Tasmax,
    Tasmin,
)
from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import VariableInfo, get_var_info

if TYPE_CHECKING:
    from pathlib import Path


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "Amon", "cl")
    assert fix == [Cl(None), GenericFix(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, BaseCl)


def test_cl_fix_file(tmp_path: Path, test_data_path: Path) -> None:
    """Test ``fix_file`` for ``cl``."""
    nc_path = test_data_path / "cesm2_waccm_cl.nc"
    vardef = get_var_info("CMIP6", "Amon", "cl")
    assert isinstance(vardef, VariableInfo)
    fix = Cl(vardef)
    fixed_cubes = fix.fix_file(nc_path, tmp_path)
    assert isinstance(fixed_cubes, Sequence)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
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
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "day", "tas")
    assert fix == [Tas(None), GenericFix(None)]


def test_tas_fix():
    """Test fix for ``tas``."""
    assert Tas is BaseTas


def test_get_pr_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "day", "pr")
    assert fix == [Pr(None), GenericFix(None)]


def test_pr_fix():
    """Test fix for ``Pr``."""
    assert Pr is BasePr


def test_get_tasmin_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "day", "tasmin")
    assert fix == [Tasmin(None), GenericFix(None)]


def test_tasmin_fix():
    """Test fix for ``Tasmin``."""
    assert Tasmin is BaseTasmin


def test_get_tasmax_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "CESM2-WACCM", "day", "tasmax")
    assert fix == [Tasmax(None), GenericFix(None)]


def test_tasmax_fix():
    """Test fix for ``Tasmax``."""
    assert Tasmax is BaseTasmax
