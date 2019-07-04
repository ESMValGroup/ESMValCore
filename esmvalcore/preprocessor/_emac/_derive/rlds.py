
"""Derivation of variable `rlds`."""

import logging
from . import var_name_constraint



def derive(cubes):
    rlds_cube = cubes.extract_strict(var_name_constraint('viso_flxtbot'))-cubes.extract_strict(var_name_constraint('rad01_tradsu'))
    
    return rlds_cube

