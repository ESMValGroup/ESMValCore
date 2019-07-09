"""Derivation of variable `pr`."""
"""The variable 'pr' (precipitation) is combined from the three EMAC variables:"""
"""'aprl_ave', 'aprc_ave' and 'aprs_ave'."""
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI)"""
"""(https://c6dreq.dkrz.de/)"""

from . import var_name_constraint


def derive(cubes):
    pr_cube = cubes.extract_strict(
        var_name_constraint('aprl_ave')) + cubes.extract_strict(
            var_name_constraint('aprc_ave')) + cubes.extract_strict(
                var_name_constraint('aprs_ave'))

    return pr_cube
