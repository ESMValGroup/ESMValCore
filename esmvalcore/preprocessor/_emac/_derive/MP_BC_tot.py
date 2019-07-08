"""Derivation of sum over black carbon tracers"""

from . import var_name_constraint

def derive(cubes):
	MP_BC_tot_cube = cubes.extract_strict(var_name_constraint('MP_BC_as_ave'))+cubes.extract_strict(var_name_constraint('MP_BC_ki_ave'))+
			 cubes.extract_strict(var_name_constraint('MP_BC_ks_ave'))
	return MP_ BC_tot_cube
