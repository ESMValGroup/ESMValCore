"""Derivation of variable `pr`."""

from . import var_name_constraint

def derive(cubes):
    pr_cube = cubes.extract_strict(var_name_constraint('aprl_ave')) + cubes.extract_strict(var_name_constraint('aprc_ave')) + cubes.extract_strict(var_name_constraint('aprs_ave'))

    return pr_cube
