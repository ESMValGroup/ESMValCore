"""Derivation of variable `ANTHNT_NO_s`."""

import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	output_cube = cubes.extract_strict(var_name_constraint('ANTHNT_NO'))
	output_cube = output_cube.collapsed(['level'],iris.analysis.SUM)
  return output_cube
