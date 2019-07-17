"""Derivation of variable `sithick`.

Change zero to missing value.

"""

import numpy as np

from . import var_name_constraint


def derive(cubes):
    """Derive `sithick`."""
    sithick_cube = cubes.extract_strict(var_name_constraint('siced_ave'))
    sithick_cube.data = np.ma.masked_equal(sithick_cube.data, 0.0)
    return sithick_cube
