"""Derivation of variable `lnox`.

The variable 'lnox' (NOx production from lightning) is not a CMOR variable. It
is derived from the two EMAC variables `NOxcg_ave` and `NOxic_ave` (following
the recipe provided by P. Jöckel in a personal communication).

"""

import iris
import iris.analysis
from cf_units import Unit

from . import var_name_constraint


def derive(cubes):
    """Derive `lnox`."""
    noxcg_cube = cubes.extract_strict(var_name_constraint('NOxcg_ave'))
    noxic_cube = cubes.extract_strict(var_name_constraint('NOxic_ave'))

    # Fix units
    noxcg_cube.units = Unit('kg')
    noxic_cube.units = Unit('kg')
    noxcg_cube = noxcg_cube.collapsed(['longitude', 'latitude'],
                                      iris.analysis.SUM)
    noxic_cube = noxic_cube.collapsed(['longitude', 'latitude'],
                                      iris.analysis.SUM)

    # Calculate lnox
    timestep = noxcg_cube.attributes['GCM_timestep']
    lnox_cube = (noxcg_cube + noxic_cube) / timestep * 365 * 24 * 3600
    return lnox_cube
