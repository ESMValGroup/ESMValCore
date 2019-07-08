"""Derivation of variable `rsutcs`."""

"""The variable 'rsutcs' (TOA Outgoing Clear-Sky Shortwave Radiation) is a """ 
"""combination of three EMAC variables: 'flxsftop_ave', 'flxstop_ave' and """
"""'srad0u_ave'. """
"""(following the recipe from the DKRZ CMIP6 Data Request WebGUI) """
"""(https://c6dreq.dkrz.de/) """
"""(The DKRZ recipe calls for 'viso_flxsftop', 'viso_flxstop' and """
"""'rad01_srad0u', however, these variables are not available in the EMAC """ 
"""CMIP6 channels therefore the variables 'flxsftop_ave', 'flxstop_ave' """
"""and 'srad0u_ave' are used.) """


from . import var_name_constraint

def derive(cubes):
    rsutcs_cube = cubes.extract_strict(var_name_constraint('flxsftop_ave')) - 
                  (cubes.extract_strict(var_name_constraint('flxstop_ave')) - 
                  cubes.extract_strict(var_name_constraint('srad0u_ave')))
                  
    return rsutcs_cube
