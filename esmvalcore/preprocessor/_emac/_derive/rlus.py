"""Derivation of variable `rlus`.

The variable 'rlus' (Surface Upwelling Longwave Radiation) is stored in the
EMAC variable 'tradsu_ave', and needs to be multiplied with -1. to represent
the CMOR standard (following the recipe from the DKRZ CMIP6 Data Request
WebGUI at https://c6dreq.dkrz.de/).

"""

from . import var_name_constraint


def derive(cubes):
    """Derive `rlus`."""
    rlus_cube = -1.0 * cubes.extract_strict(var_name_constraint('tradsu_ave'))
    return rlus_cube
