"""Derivation of variable `lnox`."""

"""The variable 'lnox' (NOx production from lightning) is not a CMOR variable."""
"""It is derived from the two EMAC variables 'NOxcg_ave' and 'NOxic_ave'."""
"""(following the recipe provided by P. Jöckel in a personal communication)"""


import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	NOxcg_ave_cube = cubes.extract_strict(var_name_constraint('NOxcg_ave'))
	NOxic_ave_cube = cubes.extract_strict(var_name_constraint('NOxic_ave'))
	Noxcg_ave_sum = NOxcg_ave_cube.collapsed(['longitude','latitude'],iris.analysis.SUM)
	Noxic_ave_sum = NOxic_ave_cube.collapsed(['longitude','latitude'],iris.analysis.SUM)
	dt = NOxcg_ave_cube.attributes['GCM_timestep']
	lnox_cube = (Noxcg_ave_sum+ Noxic_ave_sum)/dt*65*24*3600*1e9
	return lnox_cube
