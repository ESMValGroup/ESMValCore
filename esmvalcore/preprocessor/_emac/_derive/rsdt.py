"""Derivation of variable `rsdt`."""

from . import var_name_constraint

def derive(cubes):
    rsdt_cube = cubes.extract_strict(var_name_constraint('flxstop'))- cubes.extract_strict(var_name_constraint('srad0u_ave'))
    return rsdt_cube
