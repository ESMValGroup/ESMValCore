"""Derivation of variable `pr`."""

"""The variable 'pr' (precipitation) is combined from the three EMAC variables:"""
"""'aprl_ave', 'aprc_ave' and 'aprs_ave' to fulfill the CMOR standard."""
"""(following)"""
"""()

from . import var_name_constraint

def derive(cubes):
    pr_cube = cubes.extract_strict(var_name_constraint('aprl_ave')) -cubes.extract_strict(var_name_constraint('aprc_ave'))-cubes.extract_strict(var_name_constraint('aprs_ave'))

    return pr_cube
