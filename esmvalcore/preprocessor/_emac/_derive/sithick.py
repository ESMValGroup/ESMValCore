"""Derivation of variable `sithick`.

Change zero to fill_value.

"""

import numpy.ma as ma

from . import var_name_constraint

def derive(cubes):
    """Derive `sithick`."""
    sithick_cube = cubes.extract_strict(var_name_constraint('siced_ave'))
    sithick_cube = ma.masked_equal(sithick_cube, 0.)
    return sithick_cube
