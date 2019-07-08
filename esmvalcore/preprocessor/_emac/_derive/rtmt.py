"""Derivation of variable `rtmt`."""

"""The variable 'rtmt' (Net Downward Flux at Top of Model) is a combination of """
"""the two EMAC variables: 'flxttop_ave' and 'flxstop_ave'. """
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI) """
"""(https://c6dreq.dkrz.de/) """

from . import var_name_constraint

def derive(cubes):
    rtmt_cube = cubes.extract_strict(var_name_constraint('flxltop_ave')) + 
                cubes.extract_strict(var_name_constraint('flxstop_ave'))
    
    return rtmt_cube

