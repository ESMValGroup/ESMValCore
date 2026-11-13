"""Test fixes for GISS-E2-1-G."""

import dask.array as da
import numpy as np
from iris.cube import Cube

from esmvalcore.cmor._fixes.fix import Fix


def test_tos_fix():
    fix = Fix.get_fixes("CMIP6", "GISS-E2-1-G", "Omon", "tos")[0]
    cube = Cube(
        da.array([274], dtype=np.float32),
        var_name="tos",
        units="degC",
    )
    (result,) = fix.fix_metadata([cube])
    assert 0.0 < result.data < 1.0
    assert result.units == "degC"
