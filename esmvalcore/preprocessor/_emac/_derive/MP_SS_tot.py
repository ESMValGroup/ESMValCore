"""Derivation of sum over all sea salt tracers"""

from . import var_name_constraint
def derive(cubes):
	MP_SS_tot_cube = cubes.extract_strict(var_name_constraint('MP_SS_as'))+cubes.extract_strict(var_name_constraint('MP_SS_cs'))+
			 cubes.extract_strict(var_name_constraint('MP_SS_ks'))
	return MP_SS_tot_cube


