"""Derivation of variable `rlut`.

The variable 'rlut' (TOA Outgoing Longwave Radiation) is stored in the EMAC
variable 'flxttop_ave', and needs to be multiplied with -1. to represent the
CMOR standard (following the recipe from the DKRZ CMIP6 Data Request WebGUI at
https://c6dreq.dkrz.de/).

"""

from . import var_name_constraint


def derive(cubes):
    """Derive `rlus`."""
    rlut_cube = -1.0 * cubes.extract_strict(var_name_constraint('flxttop_ave'))
    return rlut_cube
