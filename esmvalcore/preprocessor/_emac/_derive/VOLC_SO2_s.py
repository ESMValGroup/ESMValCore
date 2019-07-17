"""Derivation of variable `VOLC_SO2_s`.

The variable 'VOLC_SO2_s' is an EMAC variable that is used for monitoring EMAC
output. It is here integrated over all available levels (with the help of the
field 'geopot_ave'). The variable is stored in the EMAC CMIP6 channel
'import_grid'.

VOLC_SO2_s: Volcanic SO2, summed.

"""

from scipy.constants import N_A

from ._shared import integrate_vertically


def derive(cubes):
    """Derive `VOLC_SO2_s` by vertival integration."""
    molar_mass_so2 = 64.066  # g mol-1
    mass_per_molecule_so2 = molar_mass_so2 / N_A * 1e-3  # kg
    return integrate_vertically(cubes,
                                'VOLC_SO2_SO2',
                                scale_factor=mass_per_molecule_so2)
