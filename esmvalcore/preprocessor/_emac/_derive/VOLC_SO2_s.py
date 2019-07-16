"""Derivation of variable `VOLC_SO2_s`.

The variable 'VOLC_SO2_s' is an EMAC variable that is used for monitoring EMAC
output. It is here integrated over all available levels (with the help of the
field 'geopot_ave'). The variable is stored in the EMAC CMIP6 channel
'import_grid'.

VOLC_SO2_s: Aircraft NO, summed.

"""
from ._shared import integrate_vertically


def derive(cubes):
    """Derive `VOLC_SO2_s` by vertival integration."""
    return integrate_vertically(cubes, 'VOLC_SO2_SO2')
