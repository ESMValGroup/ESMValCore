"""Derivation of variable `rsut`."""
"""The variable 'rsut' (TOA Outgoing Shortwave Radiation) is stored in """
"""the EMAC variable 'srad0u_ave',"""
"""and needs to be multiplied with -1. to represent the CMOR standard."""
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI)"""
"""(https://c6dreq.dkrz.de/)"""

from . import var_name_constraint


def derive(cubes):
    rsut_cube = -1. * cubes.extract_strict(var_name_constraint('srad0u_ave'))

return rsut_cube
