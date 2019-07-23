"""Derivation of variable `TN_GHG_CO2`.

Fix wrong units in input file.

"""

from cf_units import Unit

from . import var_name_constraint


def derive(cubes):
    """Fix unit of `TN_GHG_CO2`."""
    TN_GHG_CO2_cube = cubes.extract_strict(var_name_constraint('TN_GHG_CO2'))
    TN_GHG_CO2_cube.units = Unit('1')
    return TN_GHG_CO2_cube
