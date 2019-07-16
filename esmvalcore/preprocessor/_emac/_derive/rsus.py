"""Derivation of variable `rsus`.

The variable 'rsus' (Surface Upwelling Shortwave Radiation) is stored in the
EMAC variable 'sradsu_ave', and needs to be multiplied with -1. to represent
the CMOR standard (following the recipe from the DKRZ CMIP6 Data Request
WebGUI at https://c6dreq.dkrz.de/).

"""

from . import var_name_constraint


def derive(cubes):
    """Derive `rsus`."""
    rsus_cube = -1.0 * cubes.extract_strict(var_name_constraint('sradsu_ave'))
    return rsus_cube
