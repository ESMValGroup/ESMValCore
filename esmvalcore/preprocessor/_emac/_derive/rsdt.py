"""Derivation of variable `rsdt`."""

import logging

from . import var_name_constraint


def derive(cubes):
    rsdt_cube = cubes.extract_strict(var_name_constraint('viso_flxstop'))- cubes.extract_strict(var_name_constraint('rad01_srad0u'))
    return rsdt_cube
