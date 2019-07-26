"""Derivation of variable `od550aer`.

The aerosol optical depth at 550 nm is calculated as the sum over all
levels from the EMAC variable 'aot_opt_TOT_550_total_ave', which gives the
aerosol optical depth for each model level. The EMAC variable is stored in
channel 'AERmon'.

od550aer: total aerosoloptical depth at 550 nm.

"""

from . import var_name_constraint
from ._shared import sum_over_level
from cf_units import Unit
import iris


def derive(cubes):
    """Derive `od550aer`."""
    emac_var_name = 'aot_opt_TOT_550_total_ave'
    cube = cubes.extract_strict(var_name_constraint(emac_var_name))
    # change unit from '-' to '1'
    cube.units = Unit('1')
    # convert cube to cube list (needed as input argument for sum_over_level)
    cubelist = iris.cube.CubeList([cube])
    return sum_over_level(cubelist, [emac_var_name])
