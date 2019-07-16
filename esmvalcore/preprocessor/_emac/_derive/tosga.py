"""Derivation of variable `tosga`.

Fix wrong units in input file.

"""

from cf_units import Unit

from . import var_name_constraint


def derive(cubes):
    """Derive `tosga`."""
    tosga_cube = cubes.extract_strict(var_name_constraint('tho_ave'))
    tosga_cube.units = Unit('celsius')
    return tosga_cube
