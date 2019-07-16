"""Derivation of variable `AIRC_NO_s`.

The variable 'AIRC_NO_s' is an EMAC variable that is used for monitoring EMAC
output. It is here integrated over all available levels (with the help of the
field 'geopot_ave'). The variable is stored in the EMAC CMIP6 channel
'import_grid'.

AIRC_NO_s: Aircraft NO, summed.

"""
from ._shared import integrate_vertically


def derive(cubes):
    """Derive `AIRC_NO_s` by vertival integration."""
    return integrate_vertically(cubes, 'airc_NO')
