"""Derive `pr` for EMAC."""

from . import var_name_constraint


def derive(cubes):
    """Derivation function."""
    snacl_cube = cubes.extract_strict(var_name_constraint('snacl'))
    trflw_cube = cubes.extract_strict(var_name_constraint('trflw'))
    pr_cube = snacl_cube + trflw_cube
    return pr_cube
