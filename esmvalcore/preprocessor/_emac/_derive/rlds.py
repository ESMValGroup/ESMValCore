
"""Derivation of variable `rlds`."""

from . import var_name_constraint

def derive(cubes):
    rlds_cube = cubes.extract_strict(var_name_constraint('flxtbot'))-cubes.extract_strict(var_name_constraint('tradsu_ave'))
    
    return rlds_cube

