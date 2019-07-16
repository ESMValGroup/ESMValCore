"""Derivation of variable `ANTHNT_CO_s`.

The variable 'ANTHNT_CO_s' is an EMAC variable that is used for monitoring EMAC
output. It is here summed over all available levels. The variable is stored in
the EMAC CMIP6 channel 'import_grid'.

ANTHNT_CO_s: Anthropogenic CO, summed.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `BB_CO_s`."""
    return sum_over_level(cubes, ['ANTHNT_CO'])
