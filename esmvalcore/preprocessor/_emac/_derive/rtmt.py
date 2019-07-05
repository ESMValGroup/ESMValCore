"""Derivation of variable `rtmt`."""

import iris
from . import var_name_constraint


def derive(cubes):
	flxs_cube = cubes.extract_strict(var_name_constraint('flxt_ave'))
	flxt_cube = cubes.extract_strict(var_name_constraint('flxs_ave'))
	z_coord = flxs_cube.coords(dimensions=1)
	z_coord_name= z_coord[0].name()
	rtmt_cube = flxs_cube.extract(iris.Constraint(**{z_coord_name : 1.}))+ flxt_cube.extract(iris.Constraint(**{z_coord_name : 1.}))
	return rtmt_cube
