"""Derivation of variable `SHIP_AER_BC_s`."""

import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	output_cube = cubes.extract_strict(var_name_constraint('SHIP_AER_BC'))
	output_cube = output_cube.collapsed(['level'],iris.analysis.SUM)
    return output_cube
