"""Derivation of variable `evspsbl`."""

import logging
import iris

def derive(cubes):
    evspsbl_cube = -1.* cubes.extract_strict(iris.Constraint(name='evap_ave'))
    
    return evspsbl_cube
