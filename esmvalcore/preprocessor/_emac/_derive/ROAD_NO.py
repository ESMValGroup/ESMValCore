"""Derivation of variable `ROAD_NO`.

The variable 'ROAD_NO' is an EMAC variable that is used for monitoring EMAC
output. The variable is stored in the EMAC CMIP6 channel 'import_grid'.

Fix wrong units in input file.

"""

from scipy.constants import N_A


def derive(cubes):
    """Derive `ROAD_NO`."""
    molar_mass_no2 = 46.0055  # g mol-1
    mass_per_molecule_no2 = molar_mass_no2 / N_A * 1e-3  # kg
    return mass_per_molecule_no2 * cubes
