"""Derivation of variable `clwvi`."""

from . import var_name_constraint


def derive(cubes):
    clt_cube = cubes.extract_strict(var_name_constraint('xlvi_ave'))+ cubes.extract_strict(var_name_constraint('xivi_ave'))
      return clt_cube
