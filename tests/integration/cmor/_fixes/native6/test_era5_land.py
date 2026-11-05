"""Tests for the fixes of ERA5-Land."""

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor._fixes.native6.era5 import AllVars
from esmvalcore.cmor._fixes.native6.era5_land import Snw
from esmvalcore.cmor.table import get_tables
from esmvalcore.config import CFG

CMOR_TABLES = {
    "native6": get_tables(CFG, "native6"),
}


def _snw_cube(data):
    """Create a cube with ERA5-Land ``sd`` units (m of water equivalent)."""
    time = DimCoord(
        np.arange(len(data), dtype=float),
        standard_name="time",
        units=Unit("days since 1900-01-01"),
    )
    return Cube(
        data,
        var_name="sd",
        units="m",
        dim_coords_and_dims=[(time, 0)],
    )


def test_get_snw_fix():
    """Test whether the right fixes are gathered for snw."""
    fix = Fix.get_fixes("native6", "ERA5-Land", "day", "snw")
    vardef = CMOR_TABLES["native6"].get_variable("day", "snw")
    assert fix == [Snw(vardef), AllVars(vardef), GenericFix(vardef)]


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        ([0.0, 0.001, 0.5], [0.0, 1.0, 500.0]),
        ([1.0, 2.5], [1000.0, 2500.0]),
    ],
)
def test_snw_fix_metadata(input_data, expected):
    """Test conversion of ``sd`` from m of water equivalent to kg m-2."""
    vardef = CMOR_TABLES["native6"].get_variable("day", "snw")
    cube = _snw_cube(np.array(input_data))

    cubes = Snw(vardef).fix_metadata([cube])

    assert len(cubes) == 1
    assert cubes[0].units == Unit("kg m-2")
    np.testing.assert_allclose(cubes[0].data, expected)
