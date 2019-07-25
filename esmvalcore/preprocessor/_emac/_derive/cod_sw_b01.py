"""Derivation of variable `cod_sw_b01`.

The cloud optical depth averaged over shortwave band 01 is calculated as the
sum over all levels from the EMAC variable 'tau_cld_sw_B01_ave', which gives
the cloud optical depth for each model level. The EMAC variable is stored in
channel 'Amon'.

cod_sw_b01: cloud optical depth averaged over shortwave band 01.

"""

from . import var_name_constraint
from ._shared import sum_over_level
from cf_units import Unit
import iris


def derive(cubes):
    """Derive `cod_sw_b01`."""
    emac_var_name = 'tau_cld_sw_B01_ave'
    cube = cubes.extract_strict(var_name_constraint(emac_var_name))
    # change unit from '-' to '1'
    cube.units = Unit('1')
    # convert cube to cube list (needed as input argument for sum_over_level)
    cubelist = iris.cube.CubeList([cube])
    return sum_over_level(cubelist, [emac_var_name])
