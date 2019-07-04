"""Derivation of variable `rsutcs`."""

from . import var_name_constraint

def derive(cubes):
    rsutcs_cube = cubes.extract_strict(var_name_constraint('flxsftop_ave'))- (cubes.extract_strict(var_name_constraint('flxstop_ave'))-cubes.extract_strict(var_name_constraint('srad0u_ave')))
    return rsutcs_cube
