"""Derivation of variable `rsds`."""

import logging

from . import var_name_constraint


def derive(cubes):
    rsds_cube = cubes.extract_strict(var_name_constraint('viso_flxsbot'))- cubes.extract_strict(var_name_constraint('rad01_sradsu'))
    return rsds_cube
