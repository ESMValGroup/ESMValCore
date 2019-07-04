"""Derivation of variable `rsutcs`."""

from . import var_name_constraint

def derive(cubes):
    rsutcs_cube = cubes.extract_strict(var_name_constraint('flxsftop'))- (cubes.extract_strict(var_name_constraint('flxstop'))-cubes.extract_strict(var_name_constraint('srad0u_ave')))
    return rsutcs_cube
