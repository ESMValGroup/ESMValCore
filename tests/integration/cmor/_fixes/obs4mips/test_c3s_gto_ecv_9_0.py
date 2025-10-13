"""Test C3S-GTO-ECV-9-0 fixes."""

import numpy as np
from iris.cube import Cube

from esmvalcore.cmor.fix import fix_data
from tests import assert_array_equal


def test_fix_data_toz():
    """Test ``fix_data`` for toz."""
    cube = Cube([np.nan, 2.0])

    fixed_cube = fix_data(cube, "toz", "obs4MIPs", "C3S-GTO-ECV-9-0", "Amon")

    assert not np.isnan(fixed_cube.data).any()
    assert_array_equal(fixed_cube.data, np.ma.array([np.ma.masked, 2.0]))
