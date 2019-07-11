"""Derivation of variable `SHIP_AER_BC_s`."""
"""The variable 'SHIP_AER_BC_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is here summed over all """
"""available levels. """
"""The variable is stored in the EMAC CMIP6 channel 'import_grid'. """
"""SHIP_AER_BC_s: Ship Aerosol Black Carbon, summed """

import iris
import iris.analysis
from . import var_name_constraint


def derive(cubes):
    cube1 = cubes.extract_strict(var_name_constraint('SHIP_AER_BC'))
    z_coord = cube1.coords(dimensions=1)
    z_coord_name = z_coord[0].name()
    output_cube = cube1.collapsed(z_coord_name, iris.analysis.SUM)
    return output_cube
