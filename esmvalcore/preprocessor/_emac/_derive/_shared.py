"""Shared functions for EMAC variable derivation."""

import iris
import iris.analysis
import numpy as np

from . import var_name_constraint

IDX_LEV = 1


def sum_over_level(cubes, var_names, level_idx=1):
    """Perform sum over level coordinate."""
    cube = None
    for var_name in var_names:
        if cube is None:
            cube = cubes.extract_strict(var_name_constraint(var_name))
        else:
            cube += cubes.extract_strict(var_name_constraint(var_name))

    # Get correct coordinate
    lev_coords = cube.coords(dimensions=IDX_LEV, dim_coords=True)
    if lev_coords:
        cube.remove_coord(lev_coords[0])
    lev_coord = iris.coords.DimCoord(np.arange(cube.shape[level_idx]),
                                     var_name='level',
                                     long_name='level')
    cube.add_dim_coord(lev_coord, level_idx)

    # Sum over coordinate
    cube = cube.collapsed(lev_coord, iris.analysis.SUM)
    return cube
