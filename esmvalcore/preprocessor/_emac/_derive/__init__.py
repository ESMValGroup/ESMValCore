"""Derivation module for EMAC variables."""
import iris


def var_name_constraint(var_name):
    """:mod:`iris.Constraint` using `var_name` of an :mod:`iris.cube.Cube`."""
    return iris.Constraint(cube_func=lambda c: c.var_name == var_name)
