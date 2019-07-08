"""Derivation of variable `clt`."""

"""The variable 'clt' (total cloud fraction) is stored in the EMAC variable 'aclcov_ave',"""
"""and needs to be multiplied with 100. to represent the CMOR standard."""
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI)"""
"""(https://c6dreq.dkrz.de/)"""

from . import var_name_constraint


def derive(cubes):
    clt_cube = 100.* cubes.extract_strict(var_name_constraint('aclcov_ave'))
    
    return clt_cube
