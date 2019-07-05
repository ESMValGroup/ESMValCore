"""Derivation of variable `ANTHNT_SO2_s`."""

import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	cube1 = cubes.extract_strict(var_name_constraint('ANTHNT_SO2'))
	z_coord = cube1.coords(dimensions=1)
	z_coord_name= z_coord[0].name()
	output_cube = cube1.collapsed(z_coord_name,iris.analysis.SUM)
  	return output_cube
