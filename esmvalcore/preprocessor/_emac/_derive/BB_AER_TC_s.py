"""Derivation of variable `BB_AER_TC_s`.

The variable 'BB_AER_TC_s' is an EMAC variable that is used for monitoring EMAC
output. It is a combination/summation of the EMAC variables 'BB_AER_BC' and
'BB_AER_OC'. The latter two variables are stored in the EMAC CMIP6 channel
'import_grid'

BB_AER_TC_s: Biomass Burning CO, summed.
BB_AER_BC: Biomass Burning Aerosol Black Carbon.
BB_AER_OC: Biomass Burning Aerosol Organic Carbon.

"""

from ._shared import sum_over_level


def derive(cubes):
    """Derive `BB_AER_TC_s`."""
    return sum_over_level(cubes, ['BB_AER_BC', 'BB_AER_OC'])
