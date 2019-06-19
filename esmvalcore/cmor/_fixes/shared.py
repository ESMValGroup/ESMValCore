"""Shared functions for fixes."""
import iris
import numpy as np
from cf_units import Unit


def add_height_coord(cube):
    """Add height coordinate to cube."""
    height_coord = iris.coords.AuxCoord(2.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    cube.add_aux_coord(height_coord, ())
    return cube


def round_coordinates(cubes, decimals=5):
    """Round all dimensional coordinates of all cubes."""
    for cube in cubes:
        for coord in cube.coords(dim_coords=True):
            for attr in ('points', 'bounds'):
                setattr(coord, attr, np.round(getattr(coord, attr), decimals))
    return cubes
