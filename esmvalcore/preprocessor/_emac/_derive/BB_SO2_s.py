"""Derivation of variable `BB_SO2_s`.

The variable 'BB_SO2_s' is an EMAC variable that is used for monitoring EMAC
output. It is here summed over all available levels. The variable is stored in
the EMAC CMIP6 channel 'import_grid'.

BB_SO2_s: Biomass Burning SO2, summed.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `BB_SO2_s`."""
    return sum_over_level(cubes, ['BB_SO2'])
