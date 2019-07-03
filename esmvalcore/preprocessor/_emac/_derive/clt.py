"""Derivation of variable `clt`."""

import logging
import iris

def derive(cubes):
    clt_cube = 100.* cubes.extract_strict(iris.Constraint(name='aclcov_ave'))
    
    return clt_cube
