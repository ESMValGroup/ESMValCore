"""Derivation of variable `SHIP_NO_s`."""

"""The variable 'SHIP_NO' is an EMAC specific variable that is used """
"""as monitoring variable. It is vertically resolved, but was summed """
"""up over all levels for easier plotting. """

import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	cube1 = cubes.extract_strict(var_name_constraint('SHIP_NO'))
	z_coord = cube1.coords(dimensions=1)
	z_coord_name = z_coord[0].name()
	output_cube = cube1.collapsed(z_coord_name,iris.analysis.SUM)
	return output_cube
