"""Derivation of variable `rtmt`."""


from . import var_name_constraint

def derive(cubes):
    rtmt_cube = cubes.extract_strict(var_name_constraint('flxtbot'))[:,0,:,:]+ cubes.extract_strict(var_name_constraint('flxstop'))[:,0,:,:]
    
    return rtmt_cube

