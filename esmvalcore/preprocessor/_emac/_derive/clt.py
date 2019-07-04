"""Derivation of variable `clt`."""

from . import var_name_constraint


def derive(cubes):
    clt_cube = 100.* cubes.extract_strict(var_name_constraint('aclcov_ave'))
    
    return clt_cube
