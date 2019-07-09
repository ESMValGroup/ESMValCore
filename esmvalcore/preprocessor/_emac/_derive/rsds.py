"""Derivation of variable `rsds`."""
"""The variable 'rsds' (Surface Downwelling Shortwave Radiation) is a combination of"""
"""the two EMAC variables: 'flxsbot_ave' and 'sradsu_ave'."""
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI)"""
"""(https://c6dreq.dkrz.de/)"""

from . import var_name_constraint


def derive(cubes):
    rsds_cube = cubes.extract_strict(
        var_name_constraint('flxsbot_ave')) - cubes.extract_strict(
            var_name_constraint('sradsu_ave'))
    return rsds_cube
