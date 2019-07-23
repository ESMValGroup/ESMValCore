"""Derivation of variable `TN_GHG_CH4`.

Fix wrong units in input file.

"""

from cf_units import Unit

from . import var_name_constraint


def derive(cubes):
    """Fix unit of `TN_GHG_CH4`."""
    TN_GHG_CH4_cube = cubes.extract_strict(var_name_constraint('TN_GHG_CH4'))
    TN_GHG_CH4_cube.units = Unit('1')
    return TN_GHG_CH4_cube
