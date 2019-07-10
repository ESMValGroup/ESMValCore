"""Derivation of variable `VOLC_SO2_s`."""

"""The variable 'VOLC_SO2_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is here integrated over all """
"""available levels (with the help of the fields 'geopot_ave' """
"""and 'geosp_ave'. """
"""The variable is stored in the EMAC CMIP6 channel 'import_grid'. """
"""VOLC_SO2_s: Volcanic SO2, summed """

import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	cube1 = cubes.extract_strict(var_name_constraint('VOLC_SO2'))
	geopot_cube = cubes.extract_strict(var_name_constraint('geopot_ave'))
	geosp_cube = cubes.extract_strict(var_name_constraint('geosp_ave'))
	
	z_coord = cube1.coords(dimensions=1)
	z_coord_name = z_coord[0].name()
	output_cube = cube1.collapsed(z_coord_name,iris.analysis.SUM)
    return output_cube
