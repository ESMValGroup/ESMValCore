"""Derivation of variable `BB_NO_s`.

The variable 'BB_NO_s' is an EMAC variable that is used for monitoring EMAC
output. It is here summed over all available levels. The variable is stored in
the EMAC CMIP6 channel 'import_grid'.

BB_NO_s: Biomass Burning NO, summed.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `BB_NO_s`."""
    return sum_over_level(cubes, ['BB_NO'])
