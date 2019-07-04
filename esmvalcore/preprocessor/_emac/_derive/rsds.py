"""Derivation of variable `rsds`."""

from . import var_name_constraint


def derive(cubes):
    rsds_cube = cubes.extract_strict(var_name_constraint('flxsbot'))- cubes.extract_strict(var_name_constraint('sradsu_ave'))
    return rsds_cube
