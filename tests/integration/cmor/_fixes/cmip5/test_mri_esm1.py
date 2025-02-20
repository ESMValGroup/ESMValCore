"""Test MRI-ESM1 fixes."""

import iris.coords
import iris.cube
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip5.mri_esm1 import AllVars, Msftmyz
from esmvalcore.cmor.fix import Fix


def test_get_allvars_fix():
    assert AllVars(None) in Fix.get_fixes("CMIP5", "MRI-ESM1", "Amon", "fgco2")


@pytest.mark.parametrize("has_rotated_coord", [True, False])
def test_grid_fix(has_rotated_coord):
    """Test fix for using rotated pole grid coords instead of index coords."""
    cube = iris.cube.Cube(np.arange(4).reshape((2, 2)))
    if has_rotated_coord:
        r_coord = iris.coords.DimCoord(
            np.arange(2).astype(np.float64),
            var_name="rlat",
            standard_name="grid_latitude",
            long_name="latitude in rotated pole grid",
            units="degrees",
        )
        cube.add_dim_coord(r_coord, 0)
        h_coord = iris.coords.AuxCoord(
            np.arange(4).astype(np.float64).reshape((2, 2)),
            var_name="lat",
            standard_name="latitude",
            units="degrees",
        )
        cube.add_aux_coord(h_coord, (0, 1))

    cubes = [cube]
    for fix in Fix.get_fixes("CMIP5", "MRI-ESM1", "Omon", "fgco2"):
        cubes = fix.fix_metadata(cubes)

    assert len(cubes) == 1
    if has_rotated_coord:
        assert not cube.coords("grid_latitude")
        assert cube.coords(var_name="i")
        assert cube.coords("latitude")


def test_get_msftmyz_fix():
    assert Msftmyz(None) in Fix.get_fixes(
        "CMIP5", "MRI-ESM1", "Amon", "msftmyz"
    )
