"""Derivation of variable `rsutcs`."""

import logging

from . import var_name_constraint


def derive(cubes):
    rsutcs_cube = cubes.extract_strict(var_name_constraint('viso_flxsftop'))- (cubes.extract_strict(var_name_constraint('viso_flxstop'))-cubes.extract_strict(var_name_constraint('rad01_srad0u')))
    return rsutcs_cube
