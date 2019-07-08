"""Derivation of variable `evspsbl`."""

"""The variable 'evspsbl' (evaporation from canopy) is stored in the EMAC variable 'evap_ave',"""
"""and needs to be multiplied with -1. to represent the CMOR standard."""
"""(following the EMAC CMIP6 recipes)"""

from . import var_name_constraint

def derive(cubes):
    evspsbl_cube = -1.* cubes.extract_strict(var_name_constraint('evap_ave'))
    
    return evspsbl_cube
