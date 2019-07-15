"""Derivation of variable `SHIP_NO_s`.

The variable 'SHIP_NO_s' is an EMAC variable that is used for monitoring EMAC
output. It is here summed over all available levels. The variable is stored in
the EMAC CMIP6 channel 'import_grid'.

SHIP_NO_s: Ship NO, summed.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `SHIP_NO_s`."""
    return sum_over_level(cubes, ['SHIP_NO'])
