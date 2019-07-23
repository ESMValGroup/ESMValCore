"""Derivation of variable `rsdt`."""
"""The variable 'rsdt' (TOA Incident Shortwave Radiation) is combined from """
"""the two EMAC variables: 'flxstop_ave' and 'srad0u_ave'."""
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI)"""
"""(https://c6dreq.dkrz.de/)"""

from . import var_name_constraint


def derive(cubes):
    rsdt_cube = cubes.extract_strict(
        var_name_constraint('flxstop_ave')) - cubes.extract_strict(
            var_name_constraint('srad0u_ave'))

    return rsdt_cube

