"""Derivation of variable `evspsbl`."""

from . import var_name_constraint

def derive(cubes):
    evspsbl_cube = -1.* cubes.extract_strict(var_name_constraint('evap_ave'))
    
    return evspsbl_cube
