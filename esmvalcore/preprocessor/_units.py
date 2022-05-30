"""Metadata operations on data cubes.

Allows for unit conversions.
"""
import logging

import iris
import numpy as np

logger = logging.getLogger(__name__)


def convert_units(cube, units):
    """Convert the units of a cube to new ones.

    This converts units of a cube.

    Arguments
    ---------
        cube: iris.cube.Cube
            input cube

        units: str
            new units in udunits form

    Returns
    -------
    iris.cube.Cube
        converted cube.
    """
    cube.convert_units(units)
    return cube


def accumulate_coordinate(cube, coordinate):
    """Weight data using the bounds from a given coordinate.

    The resulting cube will then have units given by
    ``cube_units * coordinate_units``.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube for the flux

    coordinate: str
        Name of the coordinate that will be used as weights.

    Returns
    -------
    iris.cube.Cube
        Cube with the aggregated data

    Raises
    ------
    ValueError
        If the coordinate is not found in the cube.

    NotImplementedError
        If the coordinate is multidimensional.
    """
    try:
        coord = cube.coord(coordinate)
    except iris.exceptions.CoordinateNotFoundError as err:
        raise ValueError(
            "Requested coordinate %s not found in cube %s",
            coordinate, cube.summary(shorten=True)) from err

    if coord.ndim > 1:
        raise NotImplementedError(
            f'Multidimensional coordinate {coord} not supported.')

    factor = iris.coords.AuxCoord(
        np.diff(coord.bounds)[..., -1],
        var_name=coord.var_name,
        long_name=coord.long_name,
        units=coord.units,
    )
    result = cube * factor
    unit = result.units.format().split(' ')[-1]
    result.convert_units(unit)
    result.long_name = f"{cube.long_name} * {factor.long_name}"
    return result
