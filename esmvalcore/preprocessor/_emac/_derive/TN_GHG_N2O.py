"""Derivation of variable `TN_GHG_N2O`.

Fix wrong units in input file.

"""

from cf_units import Unit

from . import var_name_constraint


def derive(cubes):
    """Fix unit of `TN_GHG_N2O`."""
    TN_GHG_N2O_cube = cubes.extract_strict(var_name_constraint('TN_GHG_N2O'))
    TN_GHG_N2O_cube.units = Unit('1')
    return TN_GHG_N2O_cube
