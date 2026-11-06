"""Tests for the fixes of IPSL-CM5A2-INCA."""

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip6.ipsl_cm5a2_inca import Snw
from esmvalcore.cmor._fixes.cmip6.ipsl_cm6a_lr import AllVars
from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor.table import get_var_info


def _snw_cube(lon_points):
    """Create a ``snw`` cube with the given longitude points."""
    # Longitude is an AuxCoord because a DimCoord must be monotonic, while
    # the wrapped value 360 at the first point is not.
    lon = AuxCoord(
        np.array(lon_points, dtype=float),
        standard_name="longitude",
        var_name="lon",
        units="degrees_east",
    )
    lat = DimCoord(
        [0.0],
        standard_name="latitude",
        var_name="lat",
        units="degrees_north",
    )
    return Cube(
        np.ones((1, len(lon_points))),
        var_name="snw",
        units="kg m-2",
        dim_coords_and_dims=[(lat, 0)],
        aux_coords_and_dims=[(lon, 1)],
    )


def test_get_snw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "IPSL-CM5A2-INCA", "LImon", "snw")
    vardef = get_var_info("CMIP6", "LImon", "snw")
    assert fix == [Snw(vardef), AllVars(vardef), GenericFix(vardef)]


@pytest.mark.parametrize(
    ("lon_points", "expected"),
    [
        ([360.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
    ],
)
def test_snw_fix_metadata(lon_points, expected):
    """Test ``fix_metadata`` for ``snw`` wraps first longitude 360 to 0."""
    vardef = get_var_info("CMIP6", "LImon", "snw")
    cube = _snw_cube(lon_points)

    cubes = Snw(vardef).fix_metadata(CubeList([cube]))

    assert len(cubes) == 1
    np.testing.assert_allclose(cubes[0].coord("longitude").points, expected)
