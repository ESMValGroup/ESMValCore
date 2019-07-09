"""Derivation of variable `BB_AER_TC_s`."""

"""The variable 'BB_AER_TC_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is a combination/summation of the """
"""EMAC variables 'BB_AER_BC' and 'BB_AER_OC'. """
"""The latter two variables are stored in the EMAC CMIP6 channel """
"""'import_grid'. """
"""BB_AER_TC_s: Biomass Burning Aerosol Total Carbon, summed """
"""BB_AER_BC: Biomass Burning Aerosol Black Carbon """
"""BB_AER_OC: Biomass Burning Aerosol Organic Carbon """


import iris
import iris.analysis
from . import var_name_constraint

def derive(cubes):
	cube1 = cubes.extract_strict(var_name_constraint('BB_AER_BC'))
	cube2 = cubes.extract_strict(var_name_constraint('BB_AER_OC'))
	output = cube1 + cube2
	z_coord = cube1.coords(dimensions=1)
	z_coord_name = z_coord[0].name()
	output_cube = output_cube.collapsed(z_coord_name,iris.analysis.SUM)
    return output_cube
