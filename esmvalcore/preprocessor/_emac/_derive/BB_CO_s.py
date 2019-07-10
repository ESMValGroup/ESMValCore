"""Derivation of variable `BB_CO_s`."""
"""The variable 'BB_CO_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is here summed over all """
"""available levels. """
"""The variable is stored in the EMAC CMIP6 channel 'import_grid'. """
"""BB_CO_s: Biomass Burning CO, summed """

import iris
import iris.analysis
from . import var_name_constraint


def derive(cubes):
    cube1 = cubes.extract_strict(var_name_constraint('BB_CO'))
    z_coord = cube1.coords(dimensions=1)
    z_coord_name = z_coord[0].name()
    output_cube = cube1.collapsed(z_coord_name, iris.analysis.SUM)
    return output_cube
