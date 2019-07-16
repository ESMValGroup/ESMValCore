"""Derivation of variable `ANTHNT_AER_TC_s`.

The variable 'ANTHNT_AER_TC_s' is an EMAC variable that is used for monitoring
EMAC output. It is a combination/summation of the EMAC variables
'ANTHNT_AER_BC' and 'ANTHNT_AER_OC'. The latter two variables are stored in the
EMAC CMIP6 channel 'import_grid'.

ANTHNT_AER_TC_s: Anthropogenic Aerosol Total Carbon, summed.
ANTHNT_AER_BC: Anthropogenic Aerosol Black Carbon.
ANTHNT_AER_OC: Anthropogenic Aerosol Organic Carbon.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `BB_CO_s`."""
    return sum_over_level(cubes, ['ANTHNT_AER_BC', 'ANTHNT_AER_OC'])
